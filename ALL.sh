julia -p 40 H_FEMOCO.jl h2 | tee TEXT/L1_H2.txt
julia -p 40 H_INTERACTION.jl h2 | tee TEXT/INTERACTION_H2.txt

julia -p 40 H_FEMOCO.jl lih | tee TEXT/L1_LIH.txt
julia -p 40 H_INTERACTION.jl lih | tee TEXT/INTERACTION_LIH.txt

julia -p 40 H_FEMOCO.jl beh2 | tee TEXT/L1_BEH2.txt
julia -p 40 H_INTERACTION.jl beh2 | tee TEXT/INTERACTION_BEH2.txt

julia -p 40 H_FEMOCO.jl h2o | tee TEXT/L1_H2O.txt
julia -p 40 H_INTERACTION.jl h2o | tee TEXT/INTERACTION_H2O.txt

julia -p 40 H_FEMOCO.jl nh3 | tee TEXT/L1_NH3.txt
julia -p 40 H_INTERACTION.jl nh3 | tee TEXT/INTERACTION_NH3.txt