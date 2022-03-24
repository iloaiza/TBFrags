#FUNCTIONS FOR BUILDING TWO-BODY TENSOR FROM FRAGMENT INFO
function csa_tbt_builder(coeffs, n)
	tbt = zeros(Float64,n,n,n,n)

	idx = 1
	for i in 1:n
		for j in i:n
			tbt[i,i,j,j] = coeffs[idx]
			tbt[j,j,i,i] = coeffs[idx]
			idx += 1
		end
	end

	return tbt
end

function csa_obt_builder(coeffs, n)
	obt = zeros(Float64,n,n)

	for i in 1:n
		obt[i,i] = coeffs[i]
	end

	return obt
end

function icsa_tbt_builder(coeffs, n)
	tbt = zeros(Complex{Float64},n,n,n,n)

	idx = 1
	for i in 1:n
		for j in i:n
			tbt[i,i,j,j] = coeffs[idx]
			tbt[j,j,i,i] = coeffs[idx]
			idx += 1
		end
	end

	for i in 1:n
		for j in i:n
			tbt[i,i,j,j] += 1im * coeffs[idx]
			tbt[j,j,i,i] += 1im * coeffs[idx]
			idx += 1
		end
	end

	return tbt
end

#even coeffs for Taylor series of √(1-x²) around x=0 from x⁰ to x^(100'000) (odd powers are 0), obtained using TaylorSeries package:
#=
using TaylorSeries
t = Taylor1(Float64, 100000)
P = sqrt(1-t^2)
sqrt_taylor_coeffs = P.coeffs[1:2:end]
# =#



function sqrt_tb_taylor(coeffs; tol=1e-6, nmax = 100000)
	# return C=[α,β,γ] coeffs using Taylor expansion and arithmetico-geometric series of √(1-p²), for
	# coeffs = [a,b,c], p = cos(a)n1 + cos(b)n2 + (cos(c)-cos(a)-cos(b))n1n2 = A1 n1 + A2 n2 + A3 n1n2 
	# √(1-p²) = α n1 + β n2 + γ n1n2
	# √(1-p²) = 1 + ∑_{n=2}^{∞} A_n * (A1^n n1 + A2^n n2 + Bn n1n2)
	# where B1 = A3, B(n+1) = A1*A2 + Bn*(A1+A2+A3)
	# A_2n = sqrt_taylor_coeffs[n+1], A_(2n+1) = 0
	A1,A2,A3 = cos.(coeffs)
	C = zeros(3)
	C[1] = sqrt(1-A1^2)
	C[2] = sqrt(1-A2^2)
	A3 = A3-A1-A2

	diff = 1.0
	n = 0
	C_old = 0.0
	#Start at Bn=B2
	Bn = A3*(A1+A2+A3) + A1*A2
	while diff > tol && n < nmax
		C_old = C[3]
		n += 2
		iter = Int(n/2)
		C[3] += sqrt_taylor_coeffs[iter+1] * Bn
		Bn = Bn*(A1+A2+A3) + A1*A2
		Bn = Bn*(A1+A2+A3) + A1*A2

		diff = abs.(C[3]-C_old)
		#@show diff, n
	end

	if n == nmax
		println("Finished Taylor-series because nmax reached, convergence not achieved!")
		@show C[3], C_old, diff
	end

	return C
end

function tbpol_tbt_builder(coeffs, k, n)
	# builds tbt = p ± i√(1-p²) unitary two-body polynomial
	# k = 1 -> plus, k = 2 -> minus
	# coeffs = [a,b,c]
	# p = cos(a)n1 + cos(b)n2 + (cos(c)-cos(a)-cos(b))n1n2, construction guarantees |p|≤1
	tbt = zeros(Complex{Float64},n,n,n,n)

	if k == 1
		pm = 1
	elseif k == 2
		pm = -1
	end

	#p component:
	tbt[1,1,1,1] = coeffs[1]
	tbt[2,2,2,2] = coeffs[2]
	tbt[1,1,2,2] = coeffs[3]/2
	tbt[2,2,1,1] = coeffs[3]/2

	#√(1-p²) component:
	C = sqrt_tb_taylor(coeffs)
	tbt[1,1,1,1] += pm*1im*C[1]
	tbt[2,2,2,2] += pm*1im*C[2]
	tbt[1,1,2,2] += pm*1im*C[3]/2
	tbt[2,2,1,1] += pm*1im*C[3]/2

	return tbt
end

function cgmfr_tbt_builder(k, n)
	tbt = zeros(Float64,n,n,n,n)

	if k==1
		tbt[1,1,1,1] -= 2
	elseif k==2
		tbt[1,1,2,2] = -1
		tbt[2,2,1,1] = -1
	elseif k==3
		tbt[1,1,1,1] = -2
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
	elseif k==4
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[1,1,2,2] = 2
		tbt[2,2,1,1] = 2
	elseif k==5
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
	elseif k==6
		tbt[1,1,1,1] = -2
		tbt[3,3,2,2] = -1
		tbt[2,2,3,3] = -1
		tbt[1,1,3,3] = 1
		tbt[3,3,1,1] = 1
	elseif k==7
		tbt[1,1,1,1] = -2
		tbt[3,3,2,2] = -1
		tbt[2,2,3,3] = -1
		tbt[1,1,3,3] = 1
		tbt[3,3,1,1] = 1
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
	elseif k==8
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[1,1,3,3] = 1
		tbt[3,3,1,1] = 1
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
	elseif k==9
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[3,3,3,3] = -2
		tbt[1,1,3,3] = 1
		tbt[3,3,1,1] = 1
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
		tbt[2,2,3,3] = 1
		tbt[3,3,2,2] = 1
	elseif k==10
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[3,3,4,4] = -1
		tbt[4,4,3,3] = -1
		tbt[1,1,3,3] = 1
		tbt[3,3,1,1] = 1
		tbt[2,2,4,4] = 1
		tbt[4,4,2,2] = 1
		tbt[1,1,2,2] = 1
		tbt[2,2,1,1] = 1
	else
		error("Trying to build CGMFR class with class k=$k")
	end

	return tbt
end

function gmfr_tbt_builder(k, n)
	tbt = zeros(Float64,n,n,n,n)

	if k==1
		tbt[1,1,1,1] -= 2
	elseif k==2
		tbt[1,1,2,2] = -1
		tbt[2,2,1,1] = -1
	elseif k==3
		tbt[1,1,1,1] = -2
		tbt[2,2,2,2] = -2
		tbt[1,1,2,2] = 2
		tbt[2,2,1,1] = 2
	end

	return tbt
end

function upol_tbt_builder(ϕs, n)
	# e^(i*phi_1)n1n2 + e^(i*phi_2)(1-n1)n2 + e^(i*phi_3)n1(1-n2) + e^(i*phi_4)(1-n1)(1-n2) + h.c. 
	#h.c. part for [i,i,j,j]<->[j,j,i,i] symmetry
	tbt = zeros(Complex{Float64},n,n,n,n)

	#ϕ1 -> n1n2
	tbt[1,1,2,2] = exp(1im*ϕs[1])
	tbt[2,2,1,1] = exp(1im*ϕs[1])
	#ϕ2 -> n2 - n1n2
	tbt[1,1,2,2] -= exp(1im*ϕs[2])
	tbt[2,2,1,1] -= exp(1im*ϕs[2])
	tbt[2,2,2,2] = 2*exp(1im*ϕs[2])
	#ϕ3 -> n1 - n1n2
	tbt[1,1,2,2] -= exp(1im*ϕs[3])
	tbt[2,2,1,1] -= exp(1im*ϕs[3])
	tbt[1,1,1,1] = 2*exp(1im*ϕs[3])
	#ϕ4 -> -n1 -n2 +n1n2
	tbt[1,1,2,2] += exp(1im*ϕs[4])
	tbt[2,2,1,1] += exp(1im*ϕs[4])
	tbt[1,1,1,1] -= 2*exp(1im*ϕs[4])
	tbt[2,2,2,2] -= 2*exp(1im*ϕs[4])

	return tbt
end

function proj_tbt_builder(k, n)
	#builds projector k out of 10 for 2-body tensors
	tbt = zeros(Float64,n,n,n,n)

	if k==1
		tbt[1,1,1,1] = 1
	elseif k==2
		tbt[1,1,2,2] = 0.5
		tbt[2,2,1,1] = 0.5
	elseif k==3
		tbt[1,1,1,1] = 1
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
	elseif k==4
		tbt[1,1,1,1] = 1
		tbt[2,2,2,2] = 1
		tbt[1,1,2,2] = -1
		tbt[2,2,1,1] = -1
	elseif k==5
		tbt[1,1,1,1] = 1
		tbt[2,2,2,2] = 1
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
	elseif k==6
		tbt[1,1,1,1] = 1
		tbt[3,3,2,2] = 0.5
		tbt[2,2,3,3] = 0.5
		tbt[1,1,3,3] = -0.5
		tbt[3,3,1,1] = -0.5
	elseif k==7
		tbt[1,1,1,1] = 1
		tbt[3,3,2,2] = 0.5
		tbt[2,2,3,3] = 0.5
		tbt[1,1,3,3] = -0.5
		tbt[3,3,1,1] = -0.5
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
	elseif k==8
		tbt[1,1,1,1] = 1
		tbt[2,2,2,2] = 1
		tbt[1,1,3,3] = -0.5
		tbt[3,3,1,1] = -0.5
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
	elseif k==9
		tbt[1,1,1,1] = 1
		tbt[2,2,2,2] = 1
		tbt[3,3,3,3] = 1
		tbt[1,1,3,3] = -0.5
		tbt[3,3,1,1] = -0.5
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
		tbt[2,2,3,3] = -0.5
		tbt[3,3,2,2] = -0.5
	elseif k==10
		tbt[1,1,1,1] = 1
		tbt[2,2,2,2] = 1
		tbt[3,3,4,4] = 0.5
		tbt[4,4,3,3] = 0.5
		tbt[1,1,3,3] = -0.5
		tbt[3,3,1,1] = -0.5
		tbt[2,2,4,4] = -0.5
		tbt[4,4,2,2] = -0.5
		tbt[1,1,2,2] = -0.5
		tbt[2,2,1,1] = -0.5
	else
		error("Trying to build projector with class k=$k")
	end

	return tbt
end

function o3_map()
	pmap = zeros(Int64,2,45)
	n_ind = 1
	for k1 in 1:10
		for k2 in k1+1:10
			pmap[:,n_ind] = [k1,k2]
			n_ind += 1
		end
	end

	return pmap
end

function o3_tbt_builder(class, n; pmap=o3_map()) #90 different classes, (cgmfr(10) + cgmfr(9))/2*2. /2 comes from k1 k2 <-> k2 k1 symmetry, *2 from ±
	c45 = mod1(class, 45)
	if c45 == class
		plus = true
	else
		plus = false
	end

	k1,k2 = pmap[:,c45]
	
	if plus
		tbt = proj_tbt_builder(k1, n) + proj_tbt_builder(k2, n)
	else
		tbt = proj_tbt_builder(k1, n) - proj_tbt_builder(k2, n)
	end

	return tbt
end

function u11_tbt_builder(class, ϕs, n)
	if class == 11
		return upol_tbt_builder(ϕs, n)
	else
		return (exp(ϕs[1]) - exp(ϕs[2])) .* proj_tbt_builder(class, n)
	end
end

function ccd_tbt_builder(n)
	tbt = zeros(Complex{Float64},n,n,n,n)
	tbt[1,2,3,4] = 0.5im
	tbt[3,4,1,2] = 0.5im

	tbt[4,3,2,1] = -0.5im
	tbt[2,1,4,3] = -0.5im

	return tbt
end

function number_of_classes(flavour = META.ff :: FRAG_FLAVOUR)
	if typeof(flavour) == CGMFR
		return 10
	elseif typeof(flavour) == CSA
		return 1
	elseif typeof(flavour) == CSASD
		return 1
	elseif typeof(flavour) == iCSA
		return 1
	elseif typeof(flavour) == GMFR
		return 3
	elseif typeof(flavour) == UPOL
		return 1
	elseif typeof(flavour) == U11
		return 11
	elseif typeof(flavour) == O3
		return 90
	elseif typeof(flavour) == TBPOL
		return 2
	elseif typeof(flavour) == CCD
		return 1
	else
		error("Trying to obtain number of classes for fragment flavour $flavour, not defined!")
	end
end

function number_of_classes(flavour :: String)
	return number_of_classes(eval(Meta.parse(flavour*"()")))
end

function fragment_to_normalized_cartan_tbt(frag::fragment; frag_flavour = META.ff, u_flavour=META.uf)
	#returns obt, tbt for CSASD, otherwise just returns tbt
	if frag.spin_orb
		n = frag.n
	else
		n = Int(frag.n/2)
	end

	if typeof(frag_flavour) == CGMFR
		if n < 4
			println("ERROR: Trying to do CGMFR, requiring 4 orbitals, while number of used orbitals is $n")
			if frag.spin_orb == false
				println("Try using spin-orbitals instead of normalized orbitals by setting spin_orb=true")
			end
		end
		tbt = cgmfr_tbt_builder(frag.class, n)
	elseif typeof(frag_flavour) == CSASD
		tbt = csa_tbt_builder(frag.cn[n+1:end], n)
		obt = csa_obt_builder(frag.cn[1:n], n)
		if frag.spin_orb == false
			return obt./2, tbt ./ 4
		else
			return obt, tbt
		end
	elseif typeof(frag_flavour) == CSA
		tbt = csa_tbt_builder(frag.cn, n)
	elseif typeof(frag_flavour) == iCSA
		tbt = icsa_tbt_builder(frag.cn, n)
	elseif typeof(frag_flavour) == GMFR
		tbt = gmfr_tbt_builder(frag.class, n)
	elseif typeof(frag_flavour) == UPOL
		tbt = upol_tbt_builder(frag.cn[2:end], n)
	elseif typeof(frag_flavour) == TBPOL
		tbt = upol_tbt_builder(frag.cn[2:end], n)
	elseif typeof(frag_flavour) == CCD
		tbt = ccd_tbt_builder(n)
	elseif typeof(frag_flavour) == O3
		if n < 4
			println("ERROR: Trying to do O3, requiring 4 orbitals, while number of used orbitals is $n")
			if frag.spin_orb == false
				println("Try using spin-orbitals instead of normalized orbitals by setting spin_orb=true")
			end
		end
		tbt = o3_tbt_builder(frag.class, n)
	elseif typeof(frag_flavour) == U11
		if n < 4
			println("ERROR: Trying to do U11, requiring 4 orbitals, while number of used orbitals is $n")
			if frag.spin_orb == false
				println("Try using spin-orbitals instead of normalized orbitals by setting spin_orb=true")
			end
		end
		tbt = u11_tbt_builder(frag.class, frag.cn[2:end] , n)
	else
		error("Trying to build normalized tbt from fragment with flavour $(frag.flavour), not implemented")
	end

	if frag.spin_orb == false
		return tbt ./ 4
	else
		return tbt
	end
end


function fragment_to_normalized_tbt(frag::fragment; frag_flavour = META.ff, u_flavour=META.uf)
	if frag.spin_orb
		n = frag.n
	else
		n = Int(frag.n/2)
	end

	tbt = fragment_to_normalized_cartan_tbt(frag, frag_flavour = frag_flavour, u_flavour=u_flavour)

	if typeof(frag_flavour) == CCD
		#CCD default tbt is not a Cartan polynomial, requires generalized rotation
		return generalized_unitary_rotation(frag.u_params, tbt, n, u_flavour)
	elseif typeof(frag_flavour) == CSASD
		obt, tbt = tbt
		return unitary_SD_rotation(frag.u_params, obt, tbt, n, u_flavour)
	else
		return unitary_rotation(frag.u_params, tbt, n, u_flavour)
	end
end

function CSA_family(frag_flavour :: FRAG_FLAVOUR)
	CSA_ARR = [CSA(), iCSA(), CSASD()]

	if frag_flavour in CSA_ARR
		return true
	else
		return false
	end
end

function fragment_to_tbt(frag::fragment; frag_flavour = META.ff, u_flavour = META.uf)
	#CSASD returns obt, tbt
	if CSA_family(frag_flavour)
		#CSA family has no normalized version since all coeffs are λij's
		return fragment_to_normalized_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	else
		return frag.cn[1] * fragment_to_normalized_tbt(frag, frag_flavour=frag_flavour, u_flavour=u_flavour)
	end
end

function frag_coeff_length(n, frag_flavour = META.ff)
	# returns length(frag.cn)
	if typeof(frag_flavour) == CGMFR
		return 1
	elseif typeof(frag_flavour) == CSA
		return Int(n*(n+1)/2)
	elseif typeof(frag_flavour) == CSASD
		return Int(n + n*(n+1)/2)
	elseif typeof(frag_flavour) == iCSA
		return Int(n*(n+1))
	elseif typeof(frag_flavour) == GMFR
		return 1
	elseif typeof(frag_flavour) == UPOL
		return 5
	elseif typeof(frag_flavour) == U11
		return 5
	elseif typeof(frag_flavour) == O3
		return 1
	elseif typeof(frag_flavour) == TBPOL
		return 4
	elseif typeof(frag_flavour) == CCD
		return 1
	else
		error("Trying to get length of coefficients vector for fragment flavour $frag_flavour, not defined!")
	end
end

function frag_num_zeros(n, frag_flavour = META.ff)
	# how many of frag.cn coeffs are associated with fragment constant multiplying "special" fragment
	# in case of CSA, all coefficients are λij's for i≥j and fragment is Cartan 2nd order polynomial
	if typeof(frag_flavour) == CGMFR
		return 1
	elseif typeof(frag_flavour) == CSA
		return Int(n*(n+1)/2)
	elseif typeof(frag_flavour) == CSASD
		return Int(n + n*(n+1)/2)
	elseif typeof(frag_flavour) == iCSA
		return Int(n*(n+1))
	elseif typeof(frag_flavour) == GMFR
		return 1
	elseif typeof(frag_flavour) == UPOL
		return 1
	elseif typeof(frag_flavour) == U11
		return 1
	elseif typeof(frag_flavour) == O3
		return 1
	elseif typeof(frag_flavour) == TBPOL
		return 1
	elseif typeof(frag_flavour) == CCD
		return 1
	else
		error("Trying to get number of coefficients to be initialized in 0 for fragment flavour $frag_flavour, not defined!")
	end
end

function obt_to_tbt(obt)
	#transform one-body tensor into two-body tensor
	
    Dobt, Uobt = eigen(obt)
    #obt ≡ Uobt * Diagonal(Dobt) * (Uobt')

    n = size(obt)[1]

    tbt = zeros(n,n,n,n)
    for i in 1:n
        tbt[i,i,i,i] = Dobt[i]
    end

    rotated_tbt = zeros(n,n,n,n)

    @einsum rotated_tbt[a,b,c,d] = Uobt[a,l] * Uobt[b,l] * Uobt[c,m] * Uobt[d,m] * tbt[l,l,m,m]

    return rotated_tbt
end