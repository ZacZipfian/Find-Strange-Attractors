####### Find function new version (interactive loop)  ##########
print("Program Started: \n\nLoading Packages. \n")
using DynamicalSystems
using DataFrames
using CSV
using Images
using DelimitedFiles
using Distances
using Plots

print("Packages loaded. \n")

####### Meta calibation
attractor_equation = NewEra3
attractor_name = "NewEra3"
complete_time = 100_000




##### Define Functions
function LTPD_find(X)
    Z_finder = X[90_001:100_001,:]
    minix = minimum(Z_finder[:,1])-0.05 #+0.05 for rounding errors of the floats
    miniy = minimum(Z_finder[:,2])-0.05
    x_range = (maximum(Z_finder[:,1])+0.05) - minix
    divider_x = x_range / 100
    y_range = (maximum(Z_finder[:,2])+0.05) - miniy
    divider_y = y_range / 100
    counter_xy = 0;
    
    for i in 1:100
        x_filter = [minix + divider_x*(i-1), minix + divider_x*(i)]
        intr_y = Int.(x_filter[1] .< Z_finder[:,1] .< x_filter[2])
        #step_1 = intr_y.*Matrix(Z_finder)
        #NEWY = filter!(x->x!=0.0,step_1[:,2])
        if sum(intr_y)>0
           step_1 = intr_y.*Matrix(Z_finder)
           Y_SET = filter!(x->x!=0.0,step_1[:,2])
           for j in 1:100
               y_filter = [miniy + divider_y*(j-1), miniy + divider_y*(j)]
               if sum(y_filter[1] .< Y_SET .< y_filter[2])>0
                  counter_xy = counter_xy + 1
               end
           end
        end
    end
    return (counter_xy/100)
end


function dynamic_kld(X_val, dynam_syst, finder_constant, time_total, param_vals)
    u0_2 = [1e-6, 1e-6]
    henon = DeterministicIteratedMap(dynam_syst, u0_2, param_vals)
    X_2, t = trajectory(henon, time_total)  
    D_vals_1 = sqrt.(colwise(SqEuclidean(),X_val[2:finder_constant,1],X_val[1:(finder_constant-1),1] ) + colwise(SqEuclidean(),X_val[2:finder_constant,2],X_val[1:(finder_constant-1),2] ))
    D_vals_2 = sqrt.(colwise(SqEuclidean(),X_2[2:finder_constant,1],X_2[1:(finder_constant-1),1] ) + colwise(SqEuclidean(),X_2[2:finder_constant,2],X_2[1:(finder_constant-1),2] ))
    D_vals_1i = D_vals_1./(sum(D_vals_1)) 
    D_vals_2i = D_vals_2./(sum(D_vals_2)) 
    KLD = kl_divergence(D_vals_1i, D_vals_2i) 
    return KLD
end


function chaotic_attractors_save12_V2(N, CSVname, params, attr_eqn, total_time=100_000, IV1=[1e-5, 1e-5], setP=9999, divP=10000, lyaMAX=10_000)
    # DataFrame that gets saved as a CSV
    df = DataFrame(lyapunov = Float64[], LTPD = Float64[], KLD = Float64[],
                   a0 = Float64[], a1 = Float64[], a2 = Float64[], a3 = Float64[], a4 = Float64[],
                   a5 = Float64[], a6 = Float64[], a7 = Float64[], a8 = Float64[], a9 = Float64[],
                   a10 = Float64[], a11 = Float64[])

    for i in 1:N
        p0 = vec(rand((-setP:setP), (1, params)) / divP)  # Random parameter vector
        try
            henon = DeterministicIteratedMap(attr_eqn, IV1, p0)
            X_1, t = trajectory(henon, total_time)  # Generate trajectory
            
            # Validate trajectory results
            check1 = abs(X_1[total_time, 2] - X_1[total_time - 1, 2])
            check2 = abs(X_1[total_time, 1] - X_1[total_time - 1, 1])
            full_check = check1 + check2
            
            if (full_check > 0.01) && (full_check < total_time)
                # Compute Lyapunov value and LTPD and KLD
                lyapunov_val = lyapunov(henon, lyaMAX)
                if lyapunov_val > 0
                    ltpd = LTPD_find(X_1)
                    # Save on computation here with this if statement
                    if ltpd > 2.0
                        KLD = dynamic_kld(X_1, attr_eqn, lyaMAX, total_time, p0)
                    else
                        KLD = 0.0
                    end
                    p1 = vcat([lyapunov_val,ltpd,KLD],p0) # Concatenate values
                    push!(df, p1)
                end
            end
        catch err
            if isa(err, DomainError)
                #println("DomainError occurred. Skipping iteration.")
                nothing
            else
                rethrow(err)  # Re-raise unexpected errors
            end
        end
    end

    # Save the DataFrame to a CSV file
    CSV.write(CSVname * ".csv", df)
end


function picture_maker(Dynamsyst, u0, p0, num_points, res, invrtd, img_name)
    attractor_map = DeterministicIteratedMap(Dynamsyst, u0, p0)
    X_values, t_values = trajectory(attractor_map, num_points)
    X_values = X_values[100:length(X_values),:]
    canvas=zeros(res,res)
    x_max=findmax(X_values[:,1])[1]; x_min=findmin(X_values[:,1])[1]
    x_max+=abs(x_max-x_min)*0.1; x_min-=abs(x_max-x_min)*0.1
    y_max=findmax(X_values[:,2])[1]; y_min=findmin(X_values[:,2])[1]
    y_max+=abs(y_max-y_min)*0.1; y_min-=abs(y_max-y_min)*0.1
    tmp1=res/(x_max-x_min); tmp2=res/(y_max-y_min); 
    for i in eachindex(X_values)
        x_pos=Int(ceil((X_values[i,1]-x_min)*tmp1))
        (x_pos==0) && (x_pos==1)
        y_pos=Int(ceil((X_values[i,2]-y_min)*tmp2))
        (y_pos==0) && (y_pos==1)
        canvas[x_pos,y_pos]+=1
    end
    gray_img = sqrt.(canvas/findmax(canvas)[1])
    if invrtd == true #white background
        gray_img = gray_img - ones(res,res)
        gray_img = abs.(gray_img)
    end
    save(string(Dynamsyst)*"_"*img_name*"("*"T_"*string(num_points)*"_R_"*string(res)*")"*".png", Gray.(gray_img))
end


function generate_stra_atr_series(DataUsed, DynamSyst, stra_atr_vec, nameadd, type_sort)
    # Only works for 12 params systems right now
    if type_sort == "ltpd"
        df_srtd = sort(DataUsed, :LTPD, rev=true) 
    elseif type_sort == "kld" 
        df_srtd = sort(DataUsed, :KLD, rev=true) 
    end
    #type_sort = "ltpd"
    end_val = 15
    try
        for i in 1:length(stra_atr_vec)
            atr_num = stra_atr_vec[i]
            par_vec = df_srtd[atr_num, 4:end_val]
            picture_maker(DynamSyst, [1e-7, 1e-7], par_vec, 20_0000_00, 1000, true, type_sort*string(atr_num)*nameadd)
        end
    catch err
        if isa(err, DomainError)
            nothing
        else
            rethrow(err)  # Re-raise unexpected errors
        end
    end
end


# Define Discrete Dynamical systems

function NewEra3(u, p, n) # here `n` is "time", but we don't use it. no data for this
    x, y = u # system state
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = p # system parameters
    xn = a0 + a1*x + a2*y*sin(y) + a3*x*y*sin(x*y) + a4*sin(x*y) + a5*x*sin(x)
    yn = a6 + a7*x  + a8*y*sin(y)  + a9*x*y*sin(x*y) + a10*sin(x*y) + a11*x*sin(x)
    return SVector(xn, yn)
end

print("Creating CSV file. \n")
# Search over 100k to find new values
chaotic_attractors_save12_V2(100_000, string(attractor_name)*"_2", 12, attractor_equation)

# Read in new CSV DataFrame
chao_data = CSV.read(string(attractor_name)*"_2.csv", DataFrame)

print_length = size(chao_data,1)
print("CSV file created: Length "*string(print_length)*"\n\n")

recc1 = round(print_length*0.01)
recc2 = round(print_length*0.05)
recc3 = round(print_length*0.03)
# Find the values yourself
print("Instructions: \n 1. Press Enter key to SKIP attractor. \n 2. Press any other key to accept attractor and generate full image.\n")
print("(Images display in a different program)\n\n")
print("(Round 1) Find interesting attractors based on LTPD values: \n\n")

print("How many values do you want to look at? (Recommendation: Between "*string(Int64(recc1))*"-"*string(Int64(recc2))*")\n")
print("Enter the number: ")
input1 = readline()

DF1 = sort(chao_data, :LTPD, rev=true)
u0k = [1e-7, 1e-7]
total_timek = 90_000
global attractor_choices1 = []

for i in 1:parse(Int64,input1)
    try
        p0k = DF1[i, 4:15] #change [1]
        henonk = DeterministicIteratedMap(attractor_equation, u0k, p0k) #change [1]
        Xk, tk = trajectory(henonk, total_timek)
        scatter_plot = scatter(Xk[1000:80_000, 1], Xk[1000:80_000, 2], markersize = 0.1)
        display(scatter_plot)
        print("\n1."*string(i)*" Keep or Disregard?: ")
        choice1 = readline()
        if choice1 == ""
            print("(skipped)")
        else
            global attractor_choices1 = vcat(attractor_choices1, i)
            print("(kept)")
        end
    catch err
        if isa(err, DomainError)
            #println("DomainError occurred. Skipping iteration.")
            nothing
        else
            rethrow(err)  # Re-raise unexpected errors
        end
    end
end

print("\n(Round 2) Find interesting attractors based on KLD values: \n")

print("How many values do you want to look at? (Recommendation: "*string(Int64(recc3))*")\n")
print("Enter the number: ")
input2 = readline()

global attractor_choices2 = []
DF2 = sort(chao_data, :KLD, rev=true)

for i in 1:parse(Int64,input2)
    try
        p0k = DF2[i, 4:15] #change [1]
        henonk = DeterministicIteratedMap(attractor_equation, u0k, p0k) #change [1]
        Xk, tk = trajectory(henonk, total_timek)
        scatter_plot = scatter(Xk[1000:80_000, 1], Xk[1000:80_000, 2], markersize = 0.1)
        display(scatter_plot)
        print("\n2."*string(i)*" Keep or Disregard?: ")
        choice1 = readline()
        if choice1 == ""
            print("(skipped)")
        else
            global attractor_choices2 = vcat(attractor_choices2, i)
            print("(kept)")
        end
    catch err
        if isa(err, DomainError)
            #println("DomainError occurred. Skipping iteration.")
            nothing
        else
            rethrow(err)  # Re-raise unexpected errors
        end
    end
end

time_est = (length(attractor_choices1)+length(attractor_choices2))*1.25
print("Starting to generate images. Time Estimate: "*string(time_est)*" minutes\n\n")

# Generate the Images
generate_stra_atr_series(chao_data, attractor_equation, attractor_choices1, "(z)", "ltpd")

print("LTPD images done, now KLD images")
generate_stra_atr_series(chao_data, attractor_equation, attractor_choices2, "(z)", "kld")

print("\n\nProgram finished.")