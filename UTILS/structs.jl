#STRUCTURES FOR FLAVOUR-AGNOSTIC IMPLEMENTATIONS

#fragment(u_params,cn,class,n,spin_orb)
struct fragment
	u_params :: Array #array of parameters for building unitary
	cn :: Array #coefficients specifying fragment. First coefficient is always constant multiplying fragment
	class :: Int64 #class of fragment inside of flavour (e.g. reflection polynomial 4 out of 10 for CGMFR flavour)
	n :: Int64 #number of qubits (= number of spin-orbitals)
	spin_orb :: Bool #spin_orb=true means spin-orbitals are considered, =false considers orbitals (works with n/2 orbitals)
end

abstract type FRAG_FLAVOUR end
abstract type U_FLAVOUR end

struct meta_flavours
	ff :: FRAG_FLAVOUR
	uf :: U_FLAVOUR
end

struct CSA <: FRAG_FLAVOUR
end

struct CSASD <: FRAG_FLAVOUR
end

struct iCSA <: FRAG_FLAVOUR
end

struct CGMFR <: FRAG_FLAVOUR
end

struct GMFR <: FRAG_FLAVOUR
end

struct UPOL <: FRAG_FLAVOUR
end

struct TBPOL <: FRAG_FLAVOUR
end

struct CCD <: FRAG_FLAVOUR
end

struct U11 <: FRAG_FLAVOUR
end

struct O3 <: FRAG_FLAVOUR
end

struct MF_real <: U_FLAVOUR
end

struct MF <: U_FLAVOUR
end

meta_string = """function meta_flavours()
if frag_flavour == "CGMFR"
	ff = CGMFR()
elseif frag_flavour == "CSA"
	ff = CSA()
elseif frag_flavour == "CSASD"
	ff = CSASD()
elseif frag_flavour == "GMFR"
	ff = GMFR()
elseif frag_flavour == "UPOL"
	ff = UPOL()
elseif frag_flavour == "TBPOL"
	ff = TBPOL()
elseif frag_flavour == "CCD"
	ff = CCD()
elseif frag_flavour == "U11"
	ff = U11()
elseif frag_flavour == "O3"
	ff = O3()
elseif frag_flavour == "iCSA"
	ff = iCSA()
else
	error("Trying to build meta frag_flavour for fast compilation for $frag_flavour, not defined in structs.jl!")
end

if u_flavour == "MF-real" || u_flavour == "MFR"
	uf = MF_real()
elseif u_flavour == "MF"
	uf = MF()
else
	error("Trying to build meta u_flavour for fast compilation for $u_flavour, not defined in structs.jl!")
end

return meta_flavours(ff,uf)
end
"""

eval(Meta.parse(meta_string))

META = meta_flavours()