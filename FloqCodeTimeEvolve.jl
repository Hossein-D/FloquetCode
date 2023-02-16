using Base.Threads
using SharedArrays
using Pkg
using Plots
using LinearAlgebra
using LinearAlgebra: dot
using Random

using FileIO
using NPZ  
using JLD 
using StatsBase
using VegaDatasets
using Distributed
using CSV, DataFrames
using QuantumClifford

#addprocs(4)
using JLD2
#directory = "/Users/hosseindehghani/Desktop/Hafezi/Codes/"
#cd(directory)

#print("size(ru2) = ", size(ru2))
#e1, e2, e3 = Lattice(2048);
#] status QuantumClifford
function convBackStab(S)
    tempS = copy(S)
    S_XZRep = stab_to_gf2(tempS)

    size1 = Int(size(S_XZRep)[1])
    size2 = size(S_XZRep)[2]

    SPhase = tempS.phases
    myRep = zeros(size1, size2+1)
    Xbits = zeros(Bool, size1, Int(size2/2))
    Zbits = zeros(Bool, size1, Int(size2/2))

    for i in 1:size1
        Xbits[i, :] = S_XZRep[i, 1:Int(size2/2)]
        Zbits[i, :] = S_XZRep[i, Int(size2/2)+1:size2]             
        myRep[i, 1:2:size2] = Zbits[i, :]
        myRep[i, 2:2:size2] = Xbits[i, :]        
        sign = 0
        if SPhase[i]==0x0
            sign = 0
        elseif SPhase[i]==0x1
            sign = 0.5
        elseif SPhase[i]==0x2            
            sign = 1
        elseif SPhase[i]==0x3
            sign = 1.5          
        end
        myRep[i, size2+1] = sign
    end   
    #println("finSConv= ", S)    
    return myRep
end



function convBackMixedState(state, Nq)
    GFull = zeros(2*Nq, 2*Nq) # No phase bit
    GStab = convBackStab(stabilizerview(state));
    GDestab = convBackStab(stabilizerview(state));
    GZLogic = convBackStab(logicalzview(state));                    
    GXLogic = convBackStab(logicalxview(state));   
    numLog = size(logicalzview(state))[1];
    r = size(stabilizerview(state))[1];
    #println("r = ", r)
    GFull[1:r, :] = copy(GStab[1:r, 1:2*Nq])
    #println("r+1 = ", r+1, ", Nq = ", Nq)
    GFull[r+1:Nq, :] = copy(GZLogic[:, 1:2*Nq])                    
    GFull[Nq+1:Nq+r, :] = copy(GDestab[:, 1:2*Nq])                                        
    GFull[Nq+r+1:2*Nq, :] = copy(GXLogic[:, 1:2*Nq])      

    GStabLogic = zeros(2*Nq-r, 2*Nq) # No phase bit            
    GStabLogic[1:Nq, :] = copy(GFull[1:Nq, 1:2*Nq])
    GStabLogic[Nq+1:2*Nq-r, :] = copy(GFull[Nq+1+r:2*Nq, 1:2*Nq])
    
    return GStabLogic, GFull, r
end

function fourbitFunc(a, b, c, d)
    if a==0 && b==0
        return 0
    end
    if a==1 && b==1
        return c-d
    end
    if a==1 && b==0
        return d*(1-2*c)
    end
    if a==0 && b==1
        return c*(2*d-1)
    end    
end


function xor(v1, v2)
    return [Int(sum(x)%2) for x in zip(v1, v2)]
end


function PauliProdShort(P1, P2, phase=false)
    if phase==true
        N = Int(round((length(P1)-1)/2))
    else
        N = Int(round((length(P1))/2))
    end
    sumG = 0
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)])
    
    return sumPauli
end

function inputFunc(a, b, c=3)
    println("c = ", c)
    if isempty(c)
        println(a)
        println("c = isempty")
    end
end

function PauliProd(P1, P2, phase=false)
    # Takes two vectorial Pauli Operators. 
    #println("length(P1) = ", length(P1))
    if phase==true
        N = Int(round((length(P1)-1)/2))
    else
        N = Int(round((length(P1))/2))
    end
    
    #println("N = ", N)
    #print('N', N)
    #print('P1[:int(2*N)]', P1[:int(2*N)])
    #print('P1 ',P1)
    #print('P2 ',P2)    
    
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)]);
    
    if phase
        r1 = P1[2*N+1];
        r2 = P2[2*N+1];
        sumG = 0;
        #print('len P1', shape(P1))
        #print("P1 P2 sumPauli = ", P1, P2, sumPauli)
        for i in 1:N
            sumG += fourbitFunc(P1[2*i-1],P1[2*i],P2[2*i-1],P2[2*i])
        end
        #println("r1 r2", r1, r2)
        #println("end of sumPauli = ", sumPauli)
        #println("((2*r1 + 2*r2 + sumG) % 4)/2 = ", ((2*r1 + 2*r2 + sumG) % 4)/2)        
    end
    if phase
        return sumPauli, abs(((2*r1 + 2*r2 + sumG) % 4)/2)
    else
        return sumPauli
    end
end

function Entropy(M, A)
    #println("inside Entropy 0")
    lengthA = A[2]
    projA = projectOut(M, A)
    
    for i in 1:size(projA)[1]
        #println("projA[$i, :] =", projA[i, :])
    end
    for i in 1:size(M)[1]
        #println("M[$i, :] =", M[i, :])
    end
    #println("Clipped A = ", clippingGauge(projA))
    rankProjA = clippingGauge(projA)
    
    SA = rankProjA - lengthA/2    
    return SA
end



function clippingGauge(A)
    # Assumption: [M] = m * n
    m = size(A)[1]
    n = size(A)[2]
    h = 1
    k = 1
    tempArray = zeros(1, n)
    #println("m, n = ", m, n)
    # echelonA: Lower Right Triangular
    while h <= m && k <= n
        #println("inside while")
        #println("A[h:m, k] = ", A[h:m, k])
        #println("findmax(A[h:m, k]) = ", findmax(A[h:m, k])[2])
        i_max = h-1+findmax(A[h:m, k])[2]
        #println("i_max = ", i_max)    
        #println("k = ", k)
        #println("A[i_max, k]", A[i_max, k])
        #argmax((A[h:m, k]), axis = None)
        if A[i_max, k] == 0
            k += 1
            #println("A[i_max, k] == 0")            
            continue            
        else
            #println("tempArray[1, :] = A[h, :]")            
            tempArray[1, :] = A[h, :]

            A[h, :] = A[i_max, :]
            A[i_max, :] = tempArray[1, :]
        end
        for i in h+1:m
            #println("for i in h+1:m]")            
            
            if A[h, k]==0
                #println("A[h, k]==0  A)
            end
            f = A[i, k]/A[h, k]
            #print('f', f)
            A[i, k] = 0
            for j in k+1:n
                #print('i, j', i, j)
                A[i,j] = abs((A[i,j] - A[h,j]*f)%2)
            end
            #print('A \n', A)
        end
        h += 1
        k += 1
    end
    echelonA = zeros(m, n)
    #println("after while")
    for i in 1:m
        echelonA[i, :] = copy(A[m-i+1, :])
    end
    #println("echelonA = ", echelonA)
    

    
    zeroRow = 0
    for i in 1:size(echelonA)[1]
        for j in 1:size(echelonA)[2]
            #println('i, j', i, j)
            if (echelonA[i, j]==0) && j != size(echelonA)[2]
                continue
            elseif (echelonA[i, j]==0) && j == size(echelonA)[2]
                #println("echelonA[i, j] = ", echelonA[i, j])
                #println("size(echelonA)[2] = ", size(echelonA)[2])
                zeroRow += 1
                break
            else
                break 
            end
        end
    end
    #println("zeroRow = ", zeroRow)
    rank = size(echelonA)[1] - zeroRow      
    # Change echelonA to upper right triangular
    upperEchA = zeros(size(echelonA))
    
    for i in 1:size(A)[1]
        upperEchA[i, :] = copy(echelonA[size(A)[1]-i+1, :])
    end
    
    #println("rank = ", rank)        
    return rank, upperEchA
end


function initProdState(N, ancilla=false)
    N = Int(N)
    if !Bool(ancilla)
        Ginit = zeros(Int8, 2*N, 2*N)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are Z operators
            Ginit[i+N, 2*i] = 1 # X destabilizers are X operators
        end

    else
        N = Int(N)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are X operators
            Ginit[i+N, 2*i] = 1 # X stabilizers are Z operators
        end
    end    
        #print('Sinit Shape \n', Sinit.shape)
    return Ginit 
end


function symplecticProd(P1, P2, phase=false)
    if phase==true
        N = Int((length(P1)-1)/2)
    else
        N = Int((length(P1))/2)
    end
    sumP = 0
    
    for i in 1:N
        #print('P1[2*i]*P2[2*i+1] + P1[2*i+1]*P2[2*i]', P1[2*i]*P2[2*i+1] + P1[2*i+1]*P2[2*i])
        #print('2i, 2i+1', 2*i, 2*i+1)
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
        #print(sumP)
    end
    
    #sumP = dot(P1[2:2:end-1],P2[1:2:end-1])+dot(P1[1:2:end-1],P2[2:2:end-1])
    return sumP%2
end



function symplecticProdShort(P1, P2, phase=false)
    if phase==true
    # The sign bit is removed.
        N = Int((length(P1)-1)/2)
    else
        N = Int((length(P1))/2)            
    end
        
    sumP = 0
    for i in 1:N
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
    end
        
    return sumP%2
end

function sumx1x2(x1, x2)
    return cx1, x1[1]+x2
end

function enlargeU_IZXY(U)
    
    N = Int(round(size(U)[1]/2))

    g = zeros(N, 2, 2, 2*N+1)
    for i in 1:Int(N)
        g[i, 2, 1, 1:2*N+1] = copy(U[2*i, 1:2*N+1])
        g[i, 1, 2, 1:2*N+1] = copy(U[2*i-1, 1:2*N+1])
        g[i, 2, 2, 1:2*N] = copy(xor(U[2*i-1, 1:2*N], U[2*i, 1:2*N]))
        prodUxUz, prodSignXSignZ = PauliProd(U[2*i-1, :], U[2*i, :])
        g[i, 2, 2, 2*N+1] = ((1 + 2*prodSignXSignZ) % 4)/2
    end
    return g
end



function xor(v1, v2)
    return [Int(sum(x)%2) for x in zip(v1, v2)]
end



function convertStabRep(S)
    N = Int((size(S)[2]-1)/2)
    #println("N inside convertStabRep = ", N)
    signVec = zeros(UInt8, N)
    for i in 1:N
        signVec[i] = 0x0
    end
    Xbits = zeros(Bool, N, N)
    Zbits = zeros(Bool, N, N)    
    signI = 0x0
    for i in 1:N
        if S[i, 2*N+1]==0
            signI = 0x0
        elseif S[i, 2*N+1]==0.5
            signI = 0x1
        elseif S[i, 2*N+1]==1
            signI = 0x2            
        elseif S[i, 2*N+1]==1.5
            signI = 0x3            
        end
        #println("signI = ", signI)
        signVec[i] = signI
        for j in 1:N
            Xbits[i, j] = Bool(S[i, 2+2*(j-1)])    
            Zbits[i, j] = Bool(S[i, 1+2*(j-1)])
        end
    end
    stab = Stabilizer(signVec,Xbits,Zbits)
    
    return stab
end


function convertPauliRep(P)
    N = Int((size(P)[2]-1)/2)
    Xbits = zeros(Bool, N)
    Zbits = zeros(Bool, N)
    
    sign = 0x0
    if P[1, 2*N+1]==0
        sign = 0x0            
    elseif P[1, 2*N+1]==0.5
        sign = 0x1
    elseif P[1, 2*N+1]==1
        sign = 0x2            
    elseif P[1, 2*N+1]==1.5
        sign = 0x3            
    end
    for j in 1:N
        Xbits[j] = Bool(P[1, 2+2*(j-1)])    
        Zbits[j] = Bool(P[1, 1+2*(j-1)])
    end
    Pauli = PauliOperator(sign,Xbits,Zbits)
    return Pauli
end


function measurePauli(g, S, r, keep_result=false, phase=false)

    # r: The number of the stabilizers
    r = N;
    if phase==true
        N = Int((size(g)[2]-1)/2);
    else
        N = Int((size(g)[2])/2);
    end
    
    orthoSymProd = zeros(N)
    ZantiCommute = zeros(size(g))
    XantiCommute = zeros(size(g))   
    Zindex = Int64[]
    Xindex = Int64[]
    Zcounter = 0
    Xcounter = 0
    #println("g = ", g)
    newSZindex1 = zeros(size(g))
    newS = zeros(Float64, size(S))
    if phase==true
        newStabX = zeros(1, 2*N+1)
        newStabZ = zeros(1, 2*N+1)
    else
        newStabX = zeros(1, 2*N)
        newStabZ = zeros(1, 2*N)
    end
    measureRes = NaN  # "measureRes = x" means measure_Result = i^{2x}
    deterministic = 0;
    for i in 1:N
        Zprod = symplecticProd(g, S[i,1:2*N])
        if Zprod == 1
            Zcounter += 1
            append!(Zindex, Int(i))
            if Zcounter == 1
                ZantiCommute[1, :] = copy(S[i,1:2*N])                
            else
                ZantiCommute = vcat(ZantiCommute, S[i,1:2*N]')
            end
        end
        #println("after Zprod == 1, ZantiCommute = ", ZantiCommute)
        Xprod = symplecticProd(g, S[i+N,1:2*N])
        if Xprod == 1    
            Xcounter += 1
            append!(Xindex, Int(i+N))
            if Xcounter == 1
                XantiCommute[1, :] = copy(S[i+N,1:2*N])
            else
                XantiCommute = vcat(XantiCommute, S[i+N,1:2*N]')
            end
        end
    end

    newS = copy(S)
    if Zcounter == 0
        #println("Z=0")
        deterministic = 1
        if phase
            plusMinusG = zeros(Int(2*N)+1)
        else
            plusMinusG = zeros(Int(2*N))
        end
        for i in 1:N
            coefX = Int(symplecticProd(g, S[i+N, 1:2*N]))
            
            tempPlusMinusGProd = PauliProd(coefX*S[i, 1:2*N], plusMinusG, phase)
            tempPlusMinusG = xor(coefX*S[i, 1:2*N], plusMinusG)
            plusMinusG[1:2*N] = copy(tempPlusMinusG)
            if phase
                plusMinusG[2*N+1] = copy(tempPlusMinusGProd[2])
            end
        end
        for i in 1:2*N 
            if plusMinusG[i] != g[i]
                #println("not equal")
                #exit()
            end
        end
        if keep_result && phase
            if plusMinusG[2*N+1] == g[2*N+1]
                measureRes = 0
                #print("equal")
            else
                #print("else equal")
                measureRes = (g[2*N+1]-plusMinusG[2*N+1])%2
            end    
            if measureRes < 0
                measureRes += 2
            end
        end

    elseif Zcounter >= 1
        for i in 2:Zcounter            
            tempProdZ = PauliProd(S[Zindex[i], :], S[Zindex[1], :], phase)
            if phase
                newStabZ[1, 1:2*N] = copy(tempProdZ[1][1:2*N]) 
                newStabZ[1, 2*N+1] = copy(tempProdZ[2])
            else
                newStabZ[1, 1:2*N] = copy(tempProdZ[1:2*N])                 
            end
            newS[Zindex[i], :] = copy(newStabZ[:])
        end
        
        
        for i in 1:Xcounter
            tempProdX = PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :], phase)
            if phase
                newStabX[1, 1:2*N] = copy(tempProdX[1][1:2*N])                
                newStabX[1, 2*N+1] = copy(tempProdX[2])                
            else
                newStabX[1, 1:2*N] = copy(tempProdX[1:2*N])                 
                newS[Xindex[i], 1:end] = copy(newStabX[1, 1:end])                
            end
        end
        newSZindex1[:] = copy(newS[Int(Zindex[1]), :])
        newS[Int(Zindex[1]), :] = copy(g)        
        randVec = rand(Float64)        
        coin = [Bool(i<0.5) for i in randVec]   
        if keep_result
            if Bool(coin[1])
                newS[Int(Zindex[1]), Int(2*N)+1] = (g[Int(2*N)+1]+1)%2  #Int((g[Int(2*N)+1]+1)%2)  
                measureRes = 1 # measureRes = x means i^{2x}; x = 1 -> m=-1
            else
                measureRes = 0 # measureRes = x means i^{2x}; x=0 -> m=0
            end
        end
        newS[Int(Zindex[1])+N, :] = copy(newSZindex1[:])            
    end
    if keep_result    
        tempMeasure = 0
        if measureRes == 0
            tempMeasure = 1 #S_z = -1
        elseif isnan(measureRes)
            tempMeasure = 0     # No measurement           
        elseif measureRes == 1.0
            tempMeasure = 2  # S_z = +1
        elseif measureRes == 1.5
            tempMeasure = 1.5  # S_z = +1
        elseif measureRes == 0.5
            tempMeasure = 0.5  # S_z = +1        
        end
        measureRes = tempMeasure        
    end

    for m in 1:N
        orthoSymProd[m] = symplecticProd(newS[m, :], newS[m+Int(N), :], phase)
    end
    
    if keep_result
        return newS, measureRes, deterministic #ZantiCommute, XantiCommute
    else 
        return newS
    end
end


function ReprMixDenM(G, L, r, elem="all")
    if elem=="zstab"
        firstind = 1
        lastind = r
    elseif elem=="xstab"        
        firstind = L+1
        lastind = L+r        
    elseif elem=="zlogic"        
        firstind = r+1
        lastind = L
    elseif elem=="xlogic"        
        firstind = L+r+1
        lastind = 2*L
    elseif elem=="all"            
        firstind = 1
        lastind = 2*L        
    end
        
    for j in firstind:lastind
        println(PauliOperator(0x0,[Bool(i) for i in G[j, 2:2:end]], [Bool(i) for i in G[j, 1:2:end]]))            
    end    
end


function ReprPauli(P, text="")
    println(text, PauliOperator(0x0,[Bool(i) for i in P[2:2:end]], [Bool(i) for i in P[1:2:end]]))                
end    


function returnPaul(p,sign=0x0)
    n=length(p)
    return PauliOperator(sign,convert.(Bool,p[1:n÷2]),convert.(Bool,p[n÷2+1:n]))
end

function randu2(fixedSign=true, t=[], LHalf=[], i=[], randIntVec=[])
    
    givenU = false

    if size(randIntVec)[1] == 0
        givenU = false
    else
        givenU = true
    end
    
    if givenU
        randInt = 1+Int(randIntVec[(t-1)*LHalf+i+1])
        #println("(t-1)*LHalf + i = ", (t-1)*LHalf + i)
    else
        randInt = 1+rand(0:719)
    end    
    if givenU!=false
        u = ru2[randInt]
    elseif givenU==false
        u=rand(ru2)
    end
    #println("size u = ", size(u, 2))
    operator = CliffordOperator(Stabilizer([returnPaul(u[:,i],0x0) for i in 1:size(u,2)])) #CNOT
    if fixedSign        
        return operator #CliffordOperator(Stabilizer([returnPaul(u[:,i],0x0) for i in 1:size(u,2)]))
    else         
        return operator #CliffordOperator(Stabilizer([returnPaul(u[:,i],rand([0x0,0x2])) for i in 1:size(u,2)]))
    end
end


function projectOut(Matrix, RegionA)
    # RegionA: The first element is the starting point of region A, 
    # and the second component is the length of region A
    # sizeA = size(RegA)[1]
    numRow = size(Matrix)[1]
    #println("numRow = ", numRow)
    #println("size(Matrix)", size(Matrix))
    #println("numRow = ", numRow)
    #println("regionA2 = ", RegionA[2])    
    projectA = zeros(Int(numRow), Int(RegionA[2]))
    #println("RegionA", RegionA)
    projectA[1:numRow, 1:RegionA[2]] = copy(Matrix[1:numRow, RegionA[1]:RegionA[1]+RegionA[2]-1])
    #println("projectA[1:numRow, 1:RegionA[2]]")
    return projectA
end


function Lattice(Nq)
    # This function returns the edges to be measured according to the lattice data including the 
    # vertices and edges of a given lattice. 
    
    edges1=[];
    edges2=[];    
    edges3=[];    
    
    return edges1, edges2, edges3
    
end

function FloquetTimeEvolve(Nq, T, Ncircuit, NeMax, Nerase=100, star=[])
    # We only use the functions from the QuantumClifford Package. 
    # Nq: Number of qubits
    # T: Time steps
    # NeMax: maximum number of erased qubits.
    # Ncircuit: Number of runnings of the Floquet circuit.
    # Nerase: Number of repetitions of random error distribution for a given number of erased qubits
    
    
    A = [1, Int(Nq)];
    println(A)
    finalEE = 0;
    GStab = zeros(Nq, 2*Nq+1);
    depth=T;
    NerrVec = collect(1:NeMax);
    NerrLen = length(NerrVec);
    rSvec = zeros(Ncircuit, NerrLen, Nerase);
    rSvec_t = zeros(NerrLen, Nerase);
    
    erasProb_t = zeros(NerrLen, Nerase);
    erasProb = zeros(Ncircuit, NerrLen, Nerase);
    
    Nsvec=zeros(Ncircuit,NerrLen,Nerase);
    Ns_t=zeros(NerrLen,Nerase);
    Nlog=zeros(Ncircuit, NerrLen, Nerase);
    Nlog_t=zeros(NerrLen,Nerase);

    # Initialize the state as |000...0>
    state = one(Stabilizer, Nq);
    r=0; # Rank of Mixed State Stabilizer
    state = one(MixedDestabilizer,r,Nq);
    keepTraj=false;
    #GStab = convBackStab(stabilizerview(state));
    #println("Mixed GStab = ", GStab)
    
    tempEnt = zeros((T, 1));
    EE = zeros((Ncircuit, T));
    #stab = zeros(Bool, 1, 2*Nq);
    # Completely Mixed State with Stabilizer: Identity
    #state = Stabilizer([0x0], stab);
    GFull = zeros(2*Nq, 2*Nq); # We ignore the last bit
    #inbounds Threads.@threads for nc in 1:Ncircuit    
    for nc in 1:Ncircuit        
        #apply!(state, Hadamard, [2*L+1])
        #GStab = convBackStab(state)        
        e1, e2, e3 = Lattice(Nq);
        for t in 1:T
            println("t in measurements = ", t);
            flush(stdout)
            if t%3==0
                # MEASURE XX OPERATORS IN THE 0TH ROUND ON THE EDGES E1                
                for j in 1:Int(Nq/2);
                    
                    xarr = zeros(Bool, Nq);
                    zarr = zeros(Bool, Nq);
                    #println("e1 = ", e1[j, :])
                    xarr[e1[j, :][1]] = 1;
                    xarr[e1[j, :][2]] = 1;
                    #print(xarr)
                    pauliOp = PauliOperator(0x0,xarr,zarr)
                    #println("pauliOp = ", pauliOp)                    
                    #state, anticomindex, result = project!(state, pauliOp, keep_result=keepTraj, phases=true)
                    state, anticomindex, result = project!(state, pauliOp, keep_result=keepTraj)
                    
                    GStab = convBackStab(stabilizerview(state));
                    numLog = size(logicalzview(state))[1];
                    r = size(stabilizerview(state))[1];
                    rank, echG = clippingGauge(GStab)
                    tempEnt[t] = Nq-rank;                     
                    
                    if isnothing(result)
                        try
                            state.phases[anticomindex] = rand([0x0,0x2])
                        catch(y)
                            if isa(y, BoundsError)
                                println("Error")

                            end
                        end
                    end
                end                                            
            elseif t%3==1
                # MEASURE YY OPERATORS IN THE 1st ROUND ON THE EDGES E2
                for j in 1:Int(Nq/2) 
                    #println("j = ", j)                    
                    xarr = zeros(Bool, Nq)
                    zarr = zeros(Bool, Nq)                
                    #println("e2 = ", e2[j, :])
                    zarr[e2[j, :][1]] = 1;
                    zarr[e2[j, :][2]] = 1;
                    #print(xarr)
                    pauliOp = PauliOperator(0x0,xarr,zarr)
                    state, anticomindex, result = project!(state, pauliOp, keep_result=keepTraj)

                    GStab = convBackStab(stabilizerview(state));
                    rank, echG = clippingGauge(GStab)
                    tempEnt[t] = Nq-rank;                     
                    if isnothing(result)
                        try
                            state.phases[anticomindex] = rand([0x0,0x2])
                        catch(y)
                            if isa(y, BoundsError)
                                print("Error")
                            end
                        end
                    end
                    
                end                                
            elseif t%3==2
                # MEASURE ZZ OPERATORS IN THE 2nd ROUND ON THE EDGES E3
                for j in 1:Int(Nq/2) 
                    #println("j = ", j)                    
                    xarr = zeros(Bool, Nq)
                    zarr = zeros(Bool, Nq)                
                    #println("e3 = ", e3[j, :])
                    xarr[e3[j, :][1]] = 1;
                    xarr[e3[j, :][2]] = 1;
                    
                    zarr[e3[j, :][1]] = 1;
                    zarr[e3[j, :][2]] = 1;
                    
                    #print(xarr)
                    pauliOp = PauliOperator(0x1,xarr,zarr)
                    state, anticomindex, result = project!(state, pauliOp, keep_result=keepTraj)
                    
                    GStab = convBackStab(stabilizerview(state));
                    rank, echG = clippingGauge(GStab)
                    tempEnt[t] = Nq-rank;                                         
                    if isnothing(result)
                        try
                            state.phases[anticomindex] = rand([0x0,0x2])
                        catch(y)
                            if isa(y, BoundsError)
                                print("Error")
                            end
                        end                            
                    end
                end                                
            end            
        end 
        GStabLogic, GFull, r = convBackMixedState(state, Nq);        
        
        println("r = ", r) # Number of stabilizers. 
        println("tempEnt = ", tempEnt);
        flush(stdout)
        EE[nc, 1:T] = copy(tempEnt[1:T]);
        
        MArr = zeros(2*Nq, 2*Nq-r);
        MArr[1:2:end, 1:2*Nq-r] = transpose(GStabLogic[1:2*Nq-r, 2:2:end]) # X elements
        MArr[1:2:end, 1:2*Nq-r] = transpose(GStabLogic[1:2*Nq-r, 1:2:end]) # Z elements
        
        ng = 0;

        # NerrVec: collect(1:NeMax);
        # NerrLen: length(NerrVec);
        # Nerase: Number of random reptitions of the code for a given number of errors
        #for cnt in 1:Nerase
        @inbounds Threads.@threads for cnt in 1:Nerase
            if cnt%10==0
                println("cnt = ", cnt)
                flush(stdout)
            end
            for nek in 1:NerrLen                        
                nerr = NerrVec[nek];
                k = Nq-r; # number of logical qubits
                Ns = r-ng; # 
                scrambleSite = Random.shuffle(collect(1:Nq)) 
                Abar = sort(scrambleSite[1:nerr]) # Sites that have been erased
                errBasis = transpose(sort(vcat(2*Abar.-1, 2*Abar)))
                
                StabLogicClmn = vcat(collect(1:Ns), collect(r+1:2*Nq-r))
                MArrErr = zeros(length(errBasis), length(StabLogicClmn))
                
                for i in 1:length(errBasis)
                    xind = errBasis[i];
                    for j in 1:length(StabLogicClmn);
                        yind = StabLogicClmn[j];
                        MArrErr[i, j] = copy(MArr[xind, yind]);
                    end
                end

                rankMErr, echelonM = clippingGauge(MArrErr);
                #if nek==NerrLen
                    #show(stdout, "text/plain", echelonM)
                    #println()
                    #println("rank = ", rankMErr)                    
                #end
                
                # indSumStab = number of trivial syndromes
                
                indSumStab = findall(x->x==0, vec(sum(echelonM[:, 1:Ns], dims=2)))
                #println("indSumStab = ", indSumStab)
                
                # indSumLog = number of checks with non-trivial logical operation
                indSumLog = findall(x->x!=0, vec(sum(echelonM[:, Ns+1:Ns+2*k], dims=2)))
                #println("indSumLog = ", indSumLog)
                
                notrSind = findall(in(indSumStab), indSumLog);
                #println("notrSind = ", notrSind)
                # rSs : number of errors with zero syndrome but non-trivial logical operation
                rSs = length(notrSind);
                #println("rSs = ", rSs)
                rSvec_t[nek,cnt] = rSs;
                erasProb_t[nek, cnt] = 2 .^ (-1.0 * rSs)
                Ns_t[nek,cnt] = Ns;
                Nlog_t[nek,cnt] = k;        
            end
            #println("rSvec_t = ", rSvec_t[nek, :])
        end
        erasProb[nc, :, :] = copy(erasProb_t);
        Nsvec[nc,:,:]=copy(Ns_t);
        rSvec[nc,:,:]=copy(rSvec_t);
        Nlog[nc,:,:]=copy(Nlog_t);
    end  
    meanErasProb = zeros(1, NeMax);
    meanErasProb[1, 1:NeMax] = mean(erasProb, dims=[1, 3]);
    println("last rSvec = ", rSvec_t[:, 1])
    #println("state = ", state);
    #save("meanEarseProb-Nq$Nq-T$T-Nc$Ncircuit-NeMax$NeMax-Nerase$Nerase.jld", "data", erasProb)
    df = DataFrame(meanErasProb)
    CSV.write("meanEarseProb-Nq$Nq-T$T-Nc$Ncircuit-NeMax$NeMax-Nerase$Nerase-Star$star.csv", df, header = false, append = true)
    return state, Nsvec, rSvec, erasProb, Nlog;
    
end
