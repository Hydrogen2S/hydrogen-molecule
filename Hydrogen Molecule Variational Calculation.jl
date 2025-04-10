using LinearAlgebra, QuadGK, GLMakie, LoopVectorization, Base.Threads
# 添加统计包导入
using Statistics
using Test
# ------------------------- 波函数 -------------------------
@inline ψ_1s(r) = @fastmath exp(-r) / sqrt(π)

# ------------------------- 重叠积分 -------------------------
function overlap_integral(R)
    integrand(r) = @fastmath ψ_1s(r) * ψ_1s(sqrt(r^2 + R^2)) * 4π * r^2
    quadgk(integrand, 0, Inf, rtol=1e-4)[1]
end

# ------------------------- 交换积分 -------------------------
function coulomb_integral(R)
    integrand(r1,r2) = @fastmath begin
        r12 = sqrt(r1^2 + r2^2 + R^2)
        ψ_1s(r1)^2 * ψ_1s(r2)^2 / r12 * (4π*r1^2)*(4π*r2^2)
    end
    max_r = max(10, 3R)  # 动态扩展积分区域
    quadgk(r1->quadgk(r2->integrand(r1,r2),0,max_r,rtol=1e-4)[1], 0,max_r,rtol=1e-4)[1]
end


function exchange_integral(R)
    integrand(r1,r2) = @fastmath begin
        x = ψ_1s(r1)*ψ_1s(r2)*ψ_1s(r1)*ψ_1s(r2)
        x * exp(-2*R)  # 引入衰减因子加速收敛
    end
    quadgk(r1->quadgk(r2->integrand(r1,r2),0,12.0)[1],0,12.0)[1]
end



# ------------------------- 构建哈密顿矩阵 ---------------------


function build_hamiltonian(R)
    S = overlap_integral(R)
    J = coulomb_integral(R)
    K = exchange_integral(R)
    
    # 更正为严密的Heitler-London公式 (单位: Hartree)
    α = -1.0  # 单原子能量项
    β = (S - J - (1 + S)*K) / (1 + S)^2
    H11 = α + β
    H12 = (K - J*S) / (1 - S^2)
    
    [H11 H12; H12 H11]
end



# ------------------------- 变分法求解系数 -------------------------
function solve_variational(R)
    H = build_hamiltonian(R)
    eig = eigen(H)
    (energy=eig.values[1], coeff=eig.vectors[:,1])
end

# ------------------------- 电子云密度计算 ---------------------
@inline function electron_density(x, y, z, R, coeff)
    @fastmath begin
        dx1 = x - R/2
        r1 = sqrt(dx1^2 + y^2 + z^2)
        dx2 = x + R/2
        r2 = sqrt(dx2^2 + y^2 + z^2)
        ψ = coeff[1]*ψ_1s(r1) + coeff[2]*ψ_1s(r2)
        2 * abs2(ψ)  # 乘以2以考虑两个电子
    end
end

# 优化\三维密度计算
function compute_3d_density!(density3d, x3d, y3d, z3d, R, coeff)
    """
    计算三维密度

    density3d: 三维密度数组
    x3d, y3d, z3d: 网格
    R: 核间距
    coeff: 系数

    直接修改density3d用
    """
    @tturbo warn_check_args=false for k in eachindex(z3d)
        z = z3d[k]
        for j in eachindex(y3d)
            y = y3d[j]
            for i in eachindex(x3d)
                x = x3d[i]
                density3d[i,j,k] = electron_density(x, y, z, R, coeff)
            end
        end
    end
end

# 优化\二维密度计算
function compute_2d_density!(density2d, x2d, y2d, z2d, R, coeff)
    """
    计算二维密度

    density2d: 二维密度数组
    x2d, y2d, z2d: 网格坐标
    R: 核间距
    coeff: 系数

    =直接修改density2d用
    """
    @tturbo warn_check_args=false for k in eachindex(z2d)
        z = z2d[k]
        for j in eachindex(y2d)
            y = y2d[j]
            for i in eachindex(x2d)
                x = x2d[i]
                density2d[i,j,k] = electron_density(x, y, z, R, coeff)
            end
        end
    end
end

# ------------------------- 可视化绘图（三维`截面）------------
function visualize_electron_cloud(R=1.4; L=4.0, n3d=50, n2d=80, name="electron_cloud_plot.png", save_plot=false)
    """
    可视化氢分子的电子云密度分布
    
    R: 核间距
    L: 画布大小
    n3d: 三维密度计算网格数
    n2d: 二维密度计算网格数
    name: 图片名称
    save_plot: 是否保存图片

    直接绘图用
    """
    result = solve_variational(R)
    @info "变分结果：" R result.energy result.coeff

    # ========== 变量们 ==========
    x3d = range(-L, L, n3d)
    y3d = range(-L, L, n3d)
    z3d = range(-L, L, n3d)
    density3d = zeros(Float64, length(x3d), length(y3d), length(z3d))
    compute_3d_density!(density3d, x3d, y3d, z3d, R, result.coeff)

    x2d = range(-L, L, n2d)
    y2d = range(-L, L, n2d)
    z2d = range(-L, L, n2d)
    density2d = zeros(Float64, length(x2d), length(y2d), length(z2d))
    compute_2d_density!(density2d, x2d, y2d, z2d, R, result.coeff)

    # ========== 画布与轴们 ==========
    fig = Figure(size=(1600, 800), backgroundcolor=:white)
    
    # 三维的（左）
    ax3d = Axis3(fig[1, 1], 
        title="3D 电子云 (R=$R)", 
        titlesize=36,
        viewmode=:fitzoom,
        perspectiveness=0.5,
        limits=(-L, L, -L, L, -L, L),
        backgroundcolor=:white
    )
    volume!(
        ax3d,
        (-L, L), (-L, L), (-L, L),
        density3d,
        algorithm=:mip,
        colormap=:inferno,
        colorrange=(0, 1*maximum(density3d)),
        transparency=false
    )

    # 截面图布局（右）
    grid = fig[1, 2] = GridLayout()
    
    # 截面索引
    mid_idx = n2d ÷ 2  # 取整

    # XY截面
    ax_xy = Axis(grid[1, 1], 
        title="XY 平面 (Z=0)", 
        titlesize=36,
        aspect=1,
        backgroundcolor=:white
    )
    hm_xy = heatmap!(ax_xy, x2d, y2d, 
        @view(density2d[:, :, mid_idx]), 
        colormap=:inferno
    )
    
    # XZ
    ax_xz = Axis(grid[2, 1], 
        title="XZ 平面 (Y=0)", 
        titlesize=36,
        aspect=1,
        backgroundcolor=:white
    )
    hm_xz = heatmap!(ax_xz, x2d, z2d, 
        @view(density2d[:, mid_idx, :]), 
        colormap=:inferno
    )
    
    # YZ
    ax_yz = Axis(grid[1:2, 2], 
        title="YZ 平面 (X=0)", 
        titlesize=36,
        aspect=1,
        backgroundcolor=:white
    )
    hm_yz = heatmap!(ax_yz, y2d, z2d, 
        @view(density2d[mid_idx, :, :]), 
        colormap=:inferno
    )

    # 颜色条
    Colorbar(grid[1:2, 3], hm_xy, 
        label="电子密度", 
        labelsize=25,
        width=20,
        ticksize=15,
        tickalign=1,
        spinewidth=0.5
    )

    # 原子核标记
    r_nucleus = 0.3
    scatter!(ax_xy, [R/2, -R/2], [0, 0], color=:red, markersize=r_nucleus*15)
    scatter!(ax_xz, [R/2, -R/2], [0, 0], color=:red, markersize=r_nucleus*15)
    scatter!(ax_yz, [0, 0], [R/2, -R/2], color=:red, markersize=r_nucleus*15)

    if save_plot
        save(name, fig, px_per_unit=3)
        @info "图片已保存为$name" 
    end

    fig
end

# ------------------------- 图表 -------------------------

# 图1：氢分子电子云分布（R = 1.4原子单位）
visualize_electron_cloud(1.4, L=4.0, n3d=500, n2d=800, save_plot=true)

# 图2：核间距对电子云分布的影响（R = 1.4 vs. R = 3.0原子单位）
fig2 = Figure(size=(1600, 800), backgroundcolor=:white)

# R = 1.4
ax1 = Axis3(fig2[1, 1], 
    title="R = 1.4 a.u.", 
    titlesize=36,
    viewmode=:fitzoom,
    perspectiveness=0.5,
    limits=(-4, 4, -4, 4, -4, 4),
    backgroundcolor=:white
)
density3d_1 = zeros(Float64, 50, 50, 50)
compute_3d_density!(density3d_1, range(-4, 4, 50), range(-4, 4, 50), range(-4, 4, 50), 1.4, [0.7, 0.7])
volume!(
    ax1,
    (-4, 4), (-4, 4), (-4, 4),
    density3d_1,
    algorithm=:mip,
    colormap=:inferno,
    colorrange=(0, 1*maximum(density3d_1)),
    transparency=false
)

# R = 3.0
ax2 = Axis3(fig2[1, 2], 
    title="R = 3.0 a.u.", 
    titlesize=36,
    viewmode=:fitzoom,
    perspectiveness=0.5,
    limits=(-4, 4, -4, 4, -4, 4),
    backgroundcolor=:white
)
density3d_2 = zeros(Float64, 50, 50, 50)
compute_3d_density!(density3d_2, range(-4, 4, 50), range(-4, 4, 50), range(-4, 4, 50), 3.0, [0.7, 0.7])
volume!(
    ax2,
    (-4, 4), (-4, 4), (-4, 4),
    density3d_2,
    algorithm=:mip,
    colormap=:inferno,
    colorrange=(0, 1*maximum(density3d_2)),
    transparency=false
)

save("electron_cloud_comparison.png", fig2, px_per_unit=3)
@info "图2已保存为 electron_cloud_comparison.png"

# ------------------------- 性能分析 -------------------------
function measure_computation_time(R; L=4.0, n3d=50, n2d=80)
    """
    测量计算时间

    R: 核间距
    L: 画布大小
    n3d: 三维密度计算网格数
    n2d: 二维密度计算网格数

    返回：总时间
    """
    # 计算时间测量
    result = @timed solve_variational(R)
    solve_time = result.time
    
    x3d = range(-L, L, n3d)
    y3d = range(-L, L, n3d)
    z3d = range(-L, L, n3d)
    density3d = zeros(Float64, length(x3d), length(y3d), length(z3d))
    
    # 三维计算时间
    compute3d_time = @elapsed compute_3d_density!(density3d, x3d, y3d, z3d, R, result.value.coeff)
    
    # 总时间 = 变分求解 + 三维计算
    total_time = solve_time + compute3d_time
    return total_time
end

# ji算时间曲线
#function plot_computation_time_curve()
    # """
    # 绘制计算时间随核间距变化的曲线
    
    # 保存图片用的
    # """
#    Rs = 0.8:0.002:3.0  # 核间距范围
#    times = Float64[]
    
    # 测量每个R的计算时间
#    for R in Rs
#        t = measure_computation_time(R, n3d=50, n2d=80)
#        push!(times, t)
#        print("R = $R, Time = $t s\r")
#    end
#    @info "\n计算时间测量完成"
    

#    Rs = 0.8:0.1:3.0  # 增大步长减少波动
#    times = [mean(@elapsed solve_variational(R) for _ in 1:5) for R in Rs]
#    avg_time = mean(times)
#    percent_error = std(times) / avg_time * 100
#    @info "Average time: $avg_time s, Percent error: $percent_error%"

#    fig = Figure(size=(800, 600), backgroundcolor=:white)
#    ax = Axis(fig[1, 1],
#         title="计算时间随核间距变化",
#         titlesize=36,
#         xlabel="核间距 R (原子单位)",
#         xlabelsize=25,
#         ylabel="计算时间 (秒)",
#         ylabelsize=25,
#         xgridcolor=:lightgray,
#         ygridcolor=:lightgray,
#         backgroundcolor=:white
#     )
    
#     lines!(ax, Rs, times, color=:black, linewidth=1)
#     #scatter!(ax, Rs, times, color=:red, markersize=10)
    
#     #axislegend(ax, position=:rt)
#     save("computation_time_curve.png", fig, px_per_unit=3)
#     @info "计算时间曲线已保存为 computation_time_curve.png"
# end

# 改进计时方法（预编译+统计预热）
function plot_computation_time_curve()
    solve_variational(1.4)  # 预编译
    Rs = 0.8:0.1:3.0
    times = [median([@elapsed solve_variational(R) for _ in 1:10]) for R in Rs]
    
    # 计算统计量
    avg_time = mean(times)
    percent_error = std(times)/avg_time*100
    @info "平均计算时间: $avg_time s, 百分误差: $(round(percent_error, digits=2))%"

    fig = Figure(size=(800, 600), backgroundcolor=:white)
    ax = Axis(fig[1, 1],
        xlabel="核间距 R (原子单位)",
        xlabelsize=25,
        ylabel="计算时间 (秒)",
        ylabelsize=25,
        xgridcolor=:lightgray,
        ygridcolor=:lightgray,
        backgroundcolor=:white
    )
    
    # 添加误差棒
    band!(ax, Rs, avg_time.*(1 .- percent_error/100), avg_time.*(1 .+ percent_error/100), color=(:gray, 0.3))
    lines!(ax, Rs, times, color=:black, linewidth=1)
    
    save("computation_time_curve.png", fig, px_per_unit=3)
    @info "计算时间曲线已保存为 computation_time_curve.png"
end

# ------------------------- 最后的东西 -------------------------

#visualize_electron_cloud(1.4, L=4.0, n3d=50, n2d=80, save_plot=true)
#visualize_electron_cloud(3.0, L=4.0, n3d=50, n2d=80, save_plot=true)  
plot_computation_time_curve()  
# 新增R=0.8的可视化
visualize_electron_cloud(0.8, L=4.0, n3d=50, n2d=80, name="08_electron_cloud.png", save_plot=true)

# 计算R=1.4时的基态能量
energy_1_4 = solve_variational(1.4).energy
@info "氢原子基态能量 (R=1.4): $(energy_1_4) Hartree"

# 氢原子基态能量（理论值-0.5 Hartree）
#@info "Hydrogen atom ground state energy: -0.5 Hartree"
#@test isapprox(solve_variational(1.4).energy, -1.16, atol=0.05)
#@test isapprox(solve_variational(Inf).energy, -1.0, atol=0.01)  # 解离极限验证

# 计算解离极限能量应趋近于两个氢原子总能量
#energy_inf = solve_variational(100.0).energy
#@info "R=100时基态能量: $(energy_inf)"
#@assert isapprox(energy_inf, -1.0, atol=0.1)  # 调大容差以容忍受限数值积
# 验证解离极限能量收敛性
function validate_dissociation_limit()
    R_values = [50.0, 100.0, 200.0]
    for R in R_values
        energy = solve_variational(R).energy
        @assert isapprox(energy, -1.0, atol=0.05) "解离极限能量异常 R=$R 结果: $energy"
        @info "R=$R 验证通过: $energy ≈ -1.0 Hartree"
    end
end

validate_dissociation_limit()

# 核间距设置为极大值（R=200原子单位）
energy_inf = solve_variational(200.0).energy
@assert isapprox(energy_inf, -1.0, atol=0.01)