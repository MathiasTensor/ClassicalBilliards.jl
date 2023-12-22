using Roots
using ForwardDiff
using LinearAlgebra
using StaticArrays
using CoordinateTransformations
using QuadGK
using GLMakie

const SV{T} = StaticArrays.SVector{2, T} where {T <: Real}
abstract type AbstractWall end

"""
Contruct an abstract wall that will be used for determining in which fundamental domain the particle is or to which one it will go - via rotation
"""
struct AbstractWallRotation{T, U} <: AbstractWall  where {T, U <: Real}
    sp::SV{T}
    ep::SV{U}
    normal_coeff::SV{Union{T,U}} # So it will be able to filter out those it cannot have a collision with
    separate_indices::SV{Int}
    equation::Function
    k::T
    c::T
    function AbstractWallRotation(sp::SV{T}, ep::SV{U}, separate_indices::SV{Int}) where {T, U <: Real}
        w = ep - sp
        normal_coeff = [-w[2], w[1]]
        k = (ep[2] - sp[2]) / (sp[1] - sp[1]) # the k in y = kx + c
        c = sp[2] - k * sp[1] # the c in y = kx + c
        y(t::Union{T,U}) = k * t + c # the resulting function
        new{T, U}(sp, ep, normal_coeff, separate_indices ,y, k, c)
    end
end

"""
Contruct an abstract wall that will be used for determining in which fundamental domain the particle is or to which one it will go - via rotation
"""
struct AbstractWallReflection{T} <: AbstractWall  where T <: Real
    sp::SV{T}
    ep::SV{T}
    normal_coeff::SV{T}
    separate_indices::SV{Int}
    equation::Function
    k::T
    c::T
    function AbstractWallReflection(sp::SV{T}, ep::SV{T}, separate_indices::SV{Int}) where T <:Real
        w = ep - sp # helper function for the normal
        normal_coeff = [-w[2], w[1]] # construct the normal coefficient
        k = (ep[2] - sp[2]) / (sp[1] - sp[1]) # the k in y = kx + c
        c = sp[2] - k * sp[1] # the c in y = kx + c
        y(t::T) = k * t + c # the resulting function
        new{T}(sp, ep, normal_coeff, separate_indices, y, k, c)
    end
end

"""
Construct a Billiard from the input curve (::Function) and the abstract walls (<:AbstractWall)
"""
struct Billiard
    elems::Array{Union{Function, AbstractWall}}
end

struct Particle{T <: Real}

    pos_cartesian::SV{T}
    vel_cartesian::SV{T}
    pos_polar::Polar{T, T} # for the rotation of a given symmetry part
    vel_polar::Polar{T, T} # for the rotation of a given symmetry part
    symmetry_domain::Int

    function Particle(pos_cartesian::SV{T}, vel_cartesian::SV{T}) where T <: Real
        pos_polar = CoordinateTransformations.PolarFromCartesian(pos_cartesian)
        vel_polar = CoordinateTransformations.PolarFromCartesian(vel_cartesian)
        new{T}(pos_cartesian, vel_cartesian, pos_polar, vel_polar, 1)
    end

end

"""
The parametrized curve will be converted to cartesian coordinates and using the function params transformed such that it will a closed curve. Between the fundamental domains abstract (non-physically colliding) walls will be constructed to keep track of domain crossings.

The parametrized curve should be given as a tuple (x(t), y(t)) with t ∈[0,1]. This leads to a type signature (x(t), y(t))::Tuple{Function, Function} ➡ SVector{2, T} where T <: Real
"""
function construct_abstract_walls(fc::Function, function_domain::Tuple{T, T}, rotation_index::Int = 1) where T <:Real

    abstractwalls = Array{AbstractWallRotation}(undef, rotation_index) # array for the abstract walls. Their number is equal to the index of the rotation group

    # Construct the rotation matrix
    rotation_matrix(n::Int) = [cos(2*pi/n) -sin(2*pi/n); sin(2*pi/n) cos(2*pi/n)]
    # Construct the affine map with no translation for the base rotation
    rotation = AffineMap(rotation_matrix, zeros(2))

    # Construct the AbstractWall equations for the fundamental curve. If there is no rotation then we return nothing
    if rotation_index <= 1
        return Nothing
    end

    # The initial Abstract Wall is pushed into the array
    push!(abstractwalls, AbstractWallRotation(SV(0,0), SV(function_domain[1], fc(function_domain[1])), SV(1, 2)))

    # We make the loop that will construct the Abstract walls
    for index in 1:rotation_index

        # We form the new coordinates via repeated application of the Affine map to the initial vectors
        for _ in index
            x = SV(function_domain[1], fc(function_domain[1]))
            x = rotation(x)
        end 

        # Abstract Wall construction and addition to the array. The final index that is "index + 1" should be 1, so we make it moddulo the index of the rotation group
        push!(abstractwalls, AbstractWallRotation(SV(0,0), x, SV(index, mod(index + 1, rotation_index))))
    end

    return abstractwalls
end

"""
From this we construct abstract walls via reflection. The passed function of the reflection line should be a linear functiuon in one variable. We therefore cosntruct the reflection via rotations as we have a function already for that
"""
function construct_abstract_walls_reflection(fc::Function, function_domain::Tuple{T,T}, line_of_reflection::Function) where T <: Real
    # form the equation determine the type of rotation
end
"""
Find the intersection between the particle and the curve.
"""
function collision_curve(fc::Function, p::Particle{T}, function_domain::Tuple{T, T})  where T <: Real

    # Time variable that will be returned in the end
    t = Inf

    # Form the function that we will search the roots for
    g(x) = fc(x) - ((p.vel_cartesian[2]/p.vel_cartesian[1]) * (x - x.pos_cartesian[1]) + p.pos_cartesian[2]) 

    # Find the roots of g (array)
    roots = Roots.find_zero(g, function_domain)
    if isempty(roots)
        return Inf, SV(zero(T), zero(T))
    end

    # For all roots in the array find the collision time
    for root in roots
        # Constructor for the collision time
        tc = dot(p.vel_cartesian, SV(root)) / dot(p.vel_cartesian, p.vel_cartesian)
        # If the time is smaller than some previous time change it's value as the collision will happen sooner
        if tc < t
            t = tc
        end
    end

    # If the collision time is infinite than just return Inf
    if t == Inf
        return Inf, SV(zero(T), zero(T))
    end

    # Return the collision time and the new cartesian coordinates
    return t, p.pos_cartesian + t * p.vel_cartesian

end

"""
Find the intersection between the particle and the Abstract Wall.
"""
function collision_abstract_wall(wall::AbstractWallRotation, p::Particle{T}, function_domain::Tuple{T, T}) where T <: Real

    # Time variable that will be returned in the end
    t = Inf, cp = SV(zero(T), zero(T))
    
    # Form the function that we will search the roots for
    g(x) = wall.equation(x) - ((p.vel_cartesian[2]/p.vel_cartesian[1]) * (x - x.pos_cartesian[1]) + p.pos_cartesian[2])

    # Find the roots of g (array)
    roots = Roots.find_zero(g, function_domain)
    if isempty(roots)
        return Inf, SV(zero(T), zero(T)) # Return initial params
    end
    
    # For all roots in the array find the collision time
    for root in roots
        # Constructor for the collision time
        tc = dot(p.vel_cartesian, SV(root)) / dot(p.vel_cartesian, p.vel_cartesian)
        # If the time is smaller than some previous time change it's value as the collision will happen sooner
        if tc < t
            t = tc
        end
    end

    # If the collision time is infinite than just return initial params
    if t == Inf
        return Inf, SV(zero(T), zero(T))
    end

    # Return the collision time and the new cartesian coordinates
    return t, p.pos_cartesian + t * p.vel_cartesian

end

"""
This function will find the smallest collision time between the particle and the curve or the abstract walls
"""
function next_collision(p::Particle{T}, bd::Billiard, function_domain::Tuple{T, T}) where T <: Real
    
    # Initial values of params
    ct, cp = Inf, SV(0.0, 0.0)
    colliding_wall::Union{AbstractWall, Missing} = missing

    # For each element in the billiard it checks the collision time and the particle position. It iteratively assigns the t to ct and checks if it is smaller than the last. The smallest time is then the time it takes for the particle to get there and the position of the collision is then returned
    for (_ , elem) in enumerate(bd.elems)

        if typeof(elem) <: AbstractWall # If the element is a subtype of AbstractWall
            t, pos = collision_abstract_wall(p, elem, function_domain)
        else # It must be the curve collision
            t, pos = collision_curve(p, elem, function_domain)
        end

        if t < ct # If the time is smaller than the previous time assign it it's new value
            colliding_wall = elem # Replace the previous wall with the new one
            ct = t
            cp = pos
        end 
    end 

    return ct, cp, colliding_wall # Returns the final collision time and the final point and the colliding wall so we know which one it collides with

end

"""
Resolves the collision between the particle and the abstract wall which will rotate the velocity and and the position vector. This can be challenging b/c based in the scalar product between the particle's velocity and the normal of the abstract wall which determines in which direction the rotation will take place
"""
function resolve_abstract_wall_collision!(p::Particle{T}, wall::AbstractWallRotation, rotation_index::Int = 1) where T <: Real
    
    # Construct the leftward rotation matrix
    rotation_matrix_left = [cos(2*pi/rotation_index) -sin(2*pi/rotation_index); sin(2*pi/rotation_index) cos(2*pi/rotation_index)]
    # Construct the affine map with no translation for the base rotation
    rotation_left = AffineMap(rotation_matrix_left, zeros(2))
    
    # Construct the rightward rotation matrix
    rotation_matrix_right = [cos(2*pi/rotation_index) sin(2*pi/rotation_index); -sin(2*pi/rotation_index) cos(2*pi/rotation_index)]
    # Construct the affine map with no translation for the base rotation
    rotation_right = AffineMap(rotation_matrix_left, zeros(2))

    # Get all the information from the abstract wall
    wall_index = wall.separate_indices

    # Rotate the velocity and the position based of this index. Care must be taken in which direction the collision takes place
    if dot(wall.normal_coeff, p.vel) > 0
        p.pos_cartesian = rotation_left(p.pos_cartesian) # Rotate to the left
        p.vel_cartesian = rotation_left(p.vel_cartesian)
        p.symmetry_domain = wall_index[1] # Change the domain index in a leftward way for the particle
    else
        p.pos_cartesian = rotation_right(p.pos_cartesian) # Rotate to the right
        p.vel_cartesian = rotation_right(p.vel_cartesian)
        p.symmetry_domain = wall_index[2] # Change the domain index in a rightward way for the particle
    end

    return p.pos_cartesian, p.vel_cartesian
end

"""
The actual collision resolution function. For this we need the derivate at a given point of the curve to contruct the normal for the specular reflection
"""
function resolve_curve_collision!(p::Particle{T}, fc::Function) where T <: Real
    
    # Finding the derivative the point of contact
    grad = ForwardDiff.gradient(fc, p.pos_cartesian[1])

    # The normal at the point is
    n = SV(-grad, 1)

    # Now plug this into the specular reflection formula
    vel_cartesian_new = p.vel_cartesian - 2 * dot(n, p.vel_cartesian) * n

    # The reflection angle for the Birkhoff coordinates
    angle_radians = acos(dot(p.vel_cartesian, vel_cartesian_new) / (norm(p.vel_cartesian) * norm(vel_cartesian_new)))

    p.vel_cartesian = vel_cartesian_new # Assign new value to particle

    return angle_radians

end

"""
Creates the position timeseries between two collisions. This is trivial for the particle with constant velocity but for a MagneticParticle it is quite non-trivial
"""
function extrapolate!(p::Particle{T}) where T <: Real
    return p.pos_cartesian
end

"""
Propagate the particle to the collision point
"""
function propagate!(p::Particle{T}, pos::SV) where T <: Real
    p.pos_cartesian = pos
end

"""
Evolve the particle for one collision (in-place). This is the base bounce map implemented into the `timeseries!` function. The `timeseries!` function will iterate this function as the number of times we want to bounce a particle
"""
function bounce!(p::Particle{T}, fc::Function, bd::Billiard, function_domain::Tuple{T, T}, rotation_index::Int = 1) where T <: Real
    domain_bounce = missing
    angles = missing
    t, cp, elem = next_collision(p, bd, function_domain)
    # determine the type of collision based on if the collision was with the AbstractWall
    if typeof(elem) <: AbstractWall
        propagate!(p, cp) # Propagate the particle to the position cp
        resolve_abstract_wall_collision!(p, elem, rotation_index) # At that position resolve the collision. This will also change the domain index of the particle depending on the params of the function
   
    else # collision was with the curve
        propagate!(p, cp) 
        angle_radians = resolve_curve_collision!(p, fc) # resolve the collision and return the specularly reflected angle
        domain_bounce = p.symmetry_domain
        angles = angle_radians
    end

    return t, cp, elem, domain_bounce, angle

end

function construct_arclength(p::Particle{T}, fc::Function, function_domain::Tuple{T, T}, cp::SV{T}) where T <: Real
    
    # derivative of the function
    df(x) = ForwardDiff.derivative(fc, x)

    # integrand for the arclength
    integrand(x) = sqrt(1+df(x)^2)

    # construct the lenght of the fundamental domain
    fundamental_domain_lenght, _ = QuadGK.quadgk(integrand, function_domain[1], function_domain[2])

    # get the domain index from the particle so we can calculate the real arclength. For this we calculate the length of the curve up to the intersection point x from cp[1]. The starting point for the integration is the last point
    length_to_xp, _ = QuadGK.quadgk(integrand, function_domain[2], cp[1])
    s = length_to_xp / fundamental_domain_lenght + (p.symmetry_domain - 1) * fundamental_domain_lenght
    return s  

end

"""
Evolve the particle in the billiard `bd` for `n` collisions
and return the position timeseries `xt, yt` along with time vector `t`. This is just the multiple application of the bounce map. After every bounce we can extrapolate the position of the particle to the new colliding Obstacle (with index i) and a time of ct
"""
function timeseries!(p::Particle{T}, fc::Function, bd::Billiard, function_domain::Tuple{T, T}, rotation_index::Int = 1, n::Int = 1) where T <: Real
    
    # Initial params of the function. Will contain all the trajectory times between bounces t, the positions x and y of all the collisions between the particle and the obstacle and the number of bounces n (iterated per parameter c that counts the bounces)
    t = [0.0]
    abstract_wall_ct = 0.0 # helper function that  will add this ficticious collision to the next real collision with the curve
    xt = [p.pos[1]]
    yt = [p.pos[2]]
    
    # For the Poincare-Birkhoff coordinates
    ξt = []
    sinθt = []

    c = 0 # Number of bounces
    while c < n

        prevpos = p.pos # for the extrapolation
        prevvel = p.vel # for the extrapolation

        ct, cp, elem, domain_bounce, angle = bounce!(p, fc, bd, function_domain, rotation_index)
        # push undefined to angles since no collision takes place
        xs, ys = extrapolate!(p)
        
        push!(sinθt, sin(angle)) # Specular reflection angle
        push!(ξt, construct_arclength(p, fc, function_domain, cp)) # Construction of the actual arclength
        
        if typeof(elem) <: AbstractWall # don't count toward bounces
            abstract_wall_ct += ct # to the last real collision add the abstract wall collision time
            continue
        end
        
        push!(t, dt + abstract_wall_ct) # if there was an abstract wall collision add this time to the fundamental domain collision
        push!(xt, xs) # Add the x point collision to the trajectory plotting array
        push!(yt, ys) # Add the y point collision to the trajectory plotting array
        
        abstract_wall_ct = 0 # reset the abstract wall collision time
        c += 1
    end
    
end

"""
Rotation helper function
"""
function construct_rotation_left(rotation_index::Int = 1)
    rotation_matrix = [cos(2*pi/rotation_index) -sin(2*pi/rotation_index); sin(2*pi/rotation_index) cos(2*pi/rotation_index)]
    return  AffineMap(rotation_matrix, zeros(2))
end

"""
Rotation helper function
"""
function construct_rotation_right(rotation_index::Int = 1)
    rotation_matrix = [cos(2*pi/rotation_index) sin(2*pi/rotation_index); -sin(2*pi/rotation_index) cos(2*pi/rotation_index)]
    return  AffineMap(rotation_matrix, zeros(2))
end

"""
Map iteration helper function
"""
function iterate_operation(f::Union{Function, AffineMap}, n_times::Int, elem)
    result = elem
    for _ in 1:n_times 
        result = f(result)
    end
    return result
end

function plot_billiard(bd::Billiard, function_domain::Tuple{T, T}, max_size::Tuple{Tuple{T, T}, Tuple{T, T}}, rotation_index::Int, N_plot_points::Int = 1000) where T <: Real
    

    
    f = GLMakie.Figure()
    ax = GLMakie.Axis(f[1, 1])

    for elem in bd.elems
        if typeof(elem) <: Function
            domain = LinRange(function_domain[1], function_domain[2], N_plot_points)
            f_domain = [elem(x) for x in domain]
            pts = zip(domain, f_domain)
            for i in 1:rotation_index
                rotated_points = [iterate_operation(construct_rotation_left(rotation_index), i, pt) for pt in pts]
                rotated_points_x = [rotated_points[i][1] for i in 1:length(domain)]
                rotated_points_y = [rotated_points[i][2] for i in 1:length(domain)]
                GLMakie.plot!(ax, rotated_points_x, rotated_points_y)
            end
        elseif typeof(elem) <: AbstractWall
            k, c = elem.k, elem.c
            GLMakie.abline!(ax, c, k, linestyle = :dash, linewidth = 2)
        end
    end
    xlims!(ax, max_size[1][1], max_size[1][2])
    ylims!(ax, max_size[2][1], max_size[2][2])
    display(f)
end

######################################### testing

function TEST(x) # triangle function
    if -1.0 <= x && x < 0.0
        return x + 1.0
    elseif x >= 0.0 && x <= 1.0
        return -x + 1.0
    else
        return 0
    end
end
#=
stuff = [TEST]
for i in 1:3 
    push!(stuff, construct_abstract_walls(TEST, (-2.5, 2.5), 3)[i])
end
=#

bd = Billiard([TEST])

plot_billiard(bd, (-1.0, 1.0), ((-3.0, 3.0), (-3.0, 3.0)), 3, 500)