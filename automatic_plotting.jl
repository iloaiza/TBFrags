using Plots
using Plots.PlotMeasures
pyplot()
vol_magnification = 2e1

FONT=font(40)
SIZE=[1780,880]
L_MARG=[15mm 0mm]
B_MARG=[10mm 0mm]


RN2 = (EXPS[1:end-1] .+1 ) .^ 2
CN = abs.(COEFFS[1:end-1])
VOLS = (VARS[1:end-1]) * vol_magnification

scatter(CN, RN2, markersize=VOLS, fillalpha=0.5)
scatter!(CN, RN2, markershape=:x, markersize=10)
plot!(title="Correlation plot for $mol_name, num_frags = $(length(CN)) \n variance metric = $(VARSUM^2)", xlabel="|c_n|",ylabel="<R_n>²", legend=false)
plot!(xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,
    legendfont=FONT,left_margin=L_MARG,bottom_margin=B_MARG, titlefont=FONT)

figname = "PLOTS/"*mol_name*"_"*frag_flavour*"_"*u_flavour
savefig(figname)

println("Plotting finished, saved fig as $figname")