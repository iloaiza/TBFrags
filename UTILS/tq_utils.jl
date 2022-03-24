tq = pyimport("tequila")

function Molecule_string(mol_name, geo)
    # Generate string for tequila Molecule construction
    if mol_name == "h2"
    	mol_string = "H 0.0 0.0 0.0\nH 0.0 0.0 $geo"
    elseif mol_name == "lih"
        mol_string = "Li 0.0 0.0 0.0\nH 0.0 0.0 $geo"
    # Giving symmetrically stretch H2O. ∠HOH = 107.6°
    elseif mol_name == "h2o"
        angle = 107.6 / 2
        angle = deg2rad(angle)
        xDistance = geo * sin(angle)
        yDistance = geo * cos(angle)
        mol_string = "O 0.0 0.0 0.0\nH -$xDistance $yDistance 0.0\nH $xDistance $yDistance 0.0"
    elseif mol_name == "n2"
        mol_string = "N 0.0 0.0 0.0\nN 0.0 0.0 $geo"
    #beh2: symmetric h bond stretch, linear molecule
    elseif mol_name == "beh2"
        mol_string = "Be 0.0 0.0 0.0\nH 0.0 0.0 -$geo\nH 0.0 0.0 $geo"
    elseif mol_name == "nh3"
    # Is there a more direct way of making three vectors with specific mutual angle?
        bondAngle = 107
        bondAngle = deg2rad(bondAngle)
        cosval = cos(bondAngle)
        sinval = sin(bondAngle)

        # The idea is second and third vector dot product is cos(angle) * geometry^2. 
        thirdyRatio = (cosval - cosval^2) / sinval
        thirdxRatio = sqrt(1 - cosval^2 - thirdyRatio^2)
        mol_string = "H 0.0 0.0 $geo\nH 0.0 $(sinval*geo) $(cosval*geo)\nH $(thirdxRatio*geo) $(thirdyRatio*geo) $(cosval*geo)\nN 0.0 0.0 0.0"
    elseif mol_name == "h4"
        R = 1.737236 #diagonal distance between center and H's, geo determines angle in degrees
        angle1 = deg2rad(geo/2)
        angle2 = deg2rad(90-geo/2)
        hor_val = 2*R*sin(angle1)
        vert_val = 2*R*sin(angle2)

        mol_string = "H 0.0 0.0 0.0\nH $hor_val 0.0 0.0\nH 0.0 $vert_val 0.0\nH $hor_val $vert_val 0.0"
    else
        error("Unknown type of hamiltonian $mol_name given")
    end

    return mol_string
end

function tq_obtain_direct_ccop(mol_name, geometry, amps_type="mp2", transformation="jordan_wigner", basis="sto-3g", spin_orb = spin_orb)
    #return two-body tensor of coupled-cluster operator, using amps_type method (e.g. mp2), not hermitized!
    #also return hamiltonian
    if spin_orb == false
        error("Running CC operator with orbitals instead of spin-orbitals, set spin-orb to true!")
    end

    geom = Molecule_string(mol_name, geometry)

    if basis == "sto3g"
        tq_basis = "sto-3g"
    else
        tq_basis = basis
    end
    molecule = tq.chemistry.Molecule(geometry = geom, basis_set=tq_basis, transformation=transformation)
    #n_elecs = molecule.n_electrons
    #n_orbs = molecule.n_orbitals
    amplitudes = molecule.compute_amplitudes(method=amps_type)
    variables = amplitudes.make_parameter_dictionary()

    ex_op = of.FermionOperator.zero()
    VALS = Float64[]
    for (fw, val) in variables
        val *= -1
        if length(fw) == 4
            fw_eff = ((2*fw[1]+1, 1), (2*fw[3], 1), (2*fw[2]+1, 0), (2*fw[4], 0))
            if fw_eff[2][1] > fw_eff[1][1]
                fw_eff = (fw_eff[2], fw_eff[1], fw_eff[3], fw_eff[4])
                val *= -1
            end
            if fw_eff[4][1] > fw_eff[3][1]
                fw_eff = (fw_eff[1], fw_eff[2], fw_eff[4], fw_eff[3])
                val *= -1
            end

            ex_op += of.FermionOperator(term=fw_eff, coefficient=val)
            push!(VALS, val)
        end
    end

    ex_herm = ex_op - of.hermitian_conjugated(ex_op)
    tbt = fermionic.get_chemist_tbt(ex_op, spin_orb=spin_orb)
    tbt_herm = fermionic.get_chemist_tbt(ex_herm, spin_orb=spin_orb)

    return ex_op, tbt, molecule, 1im .* tbt_herm
end

function tq_obtain_system(mol_name, geometry, amps_type="mp2", transformation="jordan_wigner", basis="sto-3g", spin_orb = spin_orb)
	#return two-body tensor of coupled-cluster operator, using amps_type method (e.g. mp2)
	#also return hamiltonian
	geom = Molecule_string(mol_name, geometry)

	if basis == "sto3g"
        tq_basis = "sto-3g"
    else
        tq_basis = basis
    end
    molecule = tq.chemistry.Molecule(geometry = geom, basis_set=tq_basis, transformation=transformation)
    #n_elecs = molecule.n_electrons
    #n_orbs = molecule.n_orbitals
	amplitudes = molecule.compute_amplitudes(method=amps_type)
    variables = amplitudes.make_parameter_dictionary()

    ex_op = of.FermionOperator.zero()
    for (fw, val) in variables
        val *= -1
        if length(fw) == 4
            fw_eff = ((2*fw[1]+1, 1), (2*fw[3], 1), (2*fw[2]+1, 0), (2*fw[4], 0))
            if fw_eff[2][1] > fw_eff[1][1]
                fw_eff = (fw_eff[2], fw_eff[1], fw_eff[3], fw_eff[4])
                val *= -1
            end
            if fw_eff[4][1] > fw_eff[3][1]
                fw_eff = (fw_eff[1], fw_eff[2], fw_eff[4], fw_eff[3])
                val *= -1
            end

            ex_op += of.FermionOperator(term=fw_eff, coefficient=val)
        end
    end

    ex_op -= of.hermitian_conjugated(ex_op)
    tbt = 1im .* fermionic.get_chemist_tbt(ex_op, spin_orb=spin_orb)

	return 1im * ex_op, tbt, molecule
end

function tq_obtain_individual_ops(mol_name, geometry, α_max; amps_type="mp2", transformation="jordan_wigner", basis="sto-3g", spin_orb = spin_orb)
    #return list of two-body generators of excitation op, using amps_type method (e.g. mp2)
    #also return hamiltonian
    geom = Molecule_string(mol_name, geometry)

    if basis == "sto3g"
        tq_basis = "sto-3g"
    else
        tq_basis = basis
    end
    molecule = tq.chemistry.Molecule(geometry = geom, basis_set=tq_basis, transformation=transformation)
    #n_elecs = molecule.n_electrons
    #n_orbs = molecule.n_orbitals
    amplitudes = molecule.compute_amplitudes(method=amps_type)
    variables = amplitudes.make_parameter_dictionary()

    amp_tuples = collect(keys(variables))
    amp_vals = [variables[tup] for tup in amp_tuples]

    ex_ops = []
    amps = Float64[]
    for α in 1:α_max
        if sum(abs2.(amp_vals)) == 0
            println("Only found $(α-1) non-zero amplitudes, returning those instead of $α_max")
            break
        end
        ind_max = sortperm(abs.(amp_vals))[end]
        push!(amps, amp_vals[ind_max])
        amp_vals[ind_max] = 0
        amp_max = amp_tuples[ind_max]
        @show amp_max
        ex_op = of.FermionOperator.zero()
        fw = amp_max
        if length(fw) == 4
            fw_eff = ((2*fw[1]+1, 1), (2*fw[3], 1), (2*fw[2]+1, 0), (2*fw[4], 0))
            if fw_eff[2][1] > fw_eff[1][1]
                fw_eff = (fw_eff[2], fw_eff[1], fw_eff[3], fw_eff[4])
            end
            if fw_eff[4][1] > fw_eff[3][1]
                fw_eff = (fw_eff[1], fw_eff[2], fw_eff[4], fw_eff[3])
            end

            ex_op += of.FermionOperator(term=fw_eff, coefficient=1im)
        else
            error("Got amplitude that's not two-body, not implemented!")
        end
        ex_op += of.hermitian_conjugated(ex_op)
        push!(ex_ops, ex_op)
        @show ex_op
    end

    return ex_ops, molecule
end

function unitary_to_givens(u_params, n, spin_orb; u_flavour=META.uf)
    Umat = unitary_rotation_matrix(u_params, n, u_flavour=u_flavour)
    Uop = obt_to_ferm(Umat, spin_orb, norm_ord=NORM_ORDERED)
    U_of_mat = of.get_sparse_operator(Uop)
    @show U_of_mat.toarray() == U_of_mat.todense()
    givens, V, D = of.givens_decomposition(U_of_mat.toarray())

    count = 0
    @show givens
    println("Printing V")
    display(V)
    println("\nPrinting D")
    @show D
end

function excitations_to_tq(ex_ops, transformation="jordan_wigner")
    GENS = []

    for op in ex_ops
        if transformation == "bravyi_kitaev"
            op_trans = of.bravyi_kitaev(op)
        elseif transformation == "jordan_wigner"
            op_trans = of.jordan_wigner(op)
        else
            error("Trying to transform excitation generator to tequila with transformation $transformation, not implemented!")
        end

        push!(GENS, tq.QubitHamiltonian.from_openfermion(op_trans))
    end

    return GENS
end

function frags_to_generators(FRAGS, transformation="jordan_wigner")
    GENS = [] #generators correspond to Cartan polynomials
    MFRS = [] #mean-field rotations

    for frag in FRAGS
        cartan_tbt = fragment_to_normalized_cartan_tbt(frag)
        n = length(cartan_tbt[:,1,1,1])
        Umat = 1im * unitary_generator(frag.u_params, n)
        Ugen = obt_to_ferm(Umat, frag.spin_orb)

        generator = tbt_to_ferm(cartan_tbt, frag.spin_orb)

        if transformation == "bravyi_kitaev" || transformation == "bk"
            gen_trans = of.bravyi_kitaev(generator)
            U_trans = of.bravyi_kitaev(Ugen)
        elseif transformation == "jordan_wigner" || transformation == "jw"
            gen_trans = of.jordan_wigner(generator)
            U_trans = of.jordan_wigner(Ugen)
        else
            error("Trying to perform transformation $transformation for building VQE generators, operation not defined!")
        end
        push!(GENS, tq.QubitHamiltonian.from_openfermion(gen_trans + gen_trans.identity()))
        push!(MFRS, tq.QubitHamiltonian.from_openfermion(U_trans))
    end

    return GENS, MFRS
end

function tq_run_system(molecule, FRAGS, transformation)
    H = molecule.make_hamiltonian()
    num_frags = length(FRAGS)
    # first we to initialize the hartree fock state
    println("Creating HF state")
    @time U = molecule.prepare_reference()
    
    println("Transforming fragments to tequila operators")
    @time generators, mfrs = frags_to_generators(FRAGS, transformation)

    println("Building Trotterized ansatz")

    for i in 1:length(FRAGS)
        #@show mfrs[i]
        #@show generators[i]
        U += tq.gates.Trotterized(angles = 2.0, generators=[mfrs[i]], steps=1, randomize=false)
        circuit = molecule.prepare_reference()
        wfn = tq.simulate(circuit)
        @show wfn
        circuit += tq.gates.Trotterized(angles = 2.0, generators=[mfrs[i]], steps=1, randomize=false)
        wfn = tq.simulate(circuit)
        @show wfn
        psi_hf = get_wavefunction(H.to_openfermion(), "hf", num_elecs)
        @time E,psi = eigs(sparse(psi_hf.todense()), nev=1)
        if E[1] != 1
            println("Warning, diagonalizing pure density matrix resulted in eig=$(E[1])! Using corresponding wavefunction as pure")
        end
        psi_hf = psi[1:end,1]
        Umat = mfrs[i].to_matrix()
        println("Calculating rotation with matrix exponential...")
        psi_rot = exp(-1im .* Umat) * psi_hf

        @show psi_hf
        @show psi_rot
        @show sum(abs2.(psi_rot))
        #U += tq.gates.GeneralizedRotation(generator=mfrs[i], angle=2, steps=1)
        U += tq.gates.GeneralizedRotation(generator=generators[i], angle="a$i", steps=1)
        #U += tq.gates.Trotterized(generators=generators[i], angles="a$i", steps=1, randomize_component_order=false, randomize=false)
        U += tq.gates.Trotterized(angles = 2.0, generators=[mfrs[i].dagger()], steps=1, randomize=false)
        #U += tq.gates.GeneralizedRotation(generator=mfrs[i].dagger(), angle=2, steps=1)
    end

    E = tq.ExpectationValue(H=H, U=U)
    initial_vals = Dict("a$k" => FRAGS[k].cn[1] for k in 1:num_frags)
    #initial_vals = Dict("a1" => -0.10635828670376339)

    println("Starting VQE optimziation")
    @time result = tq.minimize(objective=E, method="BFGS", initial_values=initial_vals)
    println("Final energy after VQE minimization is $(result.energy)")

    return result
end

function tq_run_iterative(molecule, ex_ops, transformation)
    H = molecule.make_hamiltonian()
    α_max = length(ex_ops)

    COEFFS = zeros(α_max)
    
    println("Transforming fragments to tequila operators")
    @time GENS = excitations_to_tq(ex_ops, transformation)

    for i in 1:length(α_max)
        println("Starting VQE cycle using $i generators:")
        U = molecule.prepare_reference()
        for α in 1:i
            U += tq.gates.Trotterized(angles = "a$α", generators=[GENS[α]], steps=1, randomize=false)
        end

        E = tq.ExpectationValue(H=H, U=U)
        initial_vals = Dict("a$k" => COEFFS[k] for k in 1:α)

        @time result = tq.minimize(objective=E, method="BFGS", initial_values=initial_vals)
    end
end