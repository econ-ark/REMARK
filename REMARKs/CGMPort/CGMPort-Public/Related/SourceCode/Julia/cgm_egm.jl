# This code is a simple Endogenous Grid Method (EGM) port in Julia of the CGM code.
# I tried to make it as similar as possible.
# No optimizations for speed are introduced in this version (but it is still quite fast compared to VFI).
# In particular, this is a terrible way to write Julia code (should introduce function barriers and proper scoping).
# Tyler Beason 2018


using Interpolations

# utility functions and related
utility(c::T1,gamma::T2) where {T1,T2<:Real} = (c^(1.0-gamma))/(1.0-gamma)
uprime(c::T1,gamma::T2) where {T1,T2<:Real} = c^(-gamma)
uprimeinv(c::T1,gamma::T2) where {T1,T2<:Real} = c^(-1.0/gamma)

# find a number in a grid
ntoi(x::Number,g::AbstractArray) = convert(Int,round((clamp(x,g[1],g[end]) - g[1])/step(g) + 1)) 
#ntoi(x::Number,g::AbstractArray) = max(searchsortedlast(g,x),1) #equivalent to above, but works with non-Range types

# parameters and such
const tb=20
const tr=65
const td=100
const nqp=3
const nalfa=201
const ns=1001
const rf=1.02
const sigma_r=0.157^2
const exc= 0.04
const delta=0.96
const gamma= 10.0

const ageprof = [-2.170042+2.700381, 0.16818, -0.0323371/10, 0.0019704/100]
const sigt_y=0.0738
const sigp_y=0.01065
const ret_fac=0.68212
const reg_coef = 0.0

const weig = [1/6; 2/3; 1/6]
const grid = [-sqrt(3); 0.0; sqrt(3)]

# SURVIVAL PROBABILITIES
survprob = Vector{Float64}(80)
survprob[1] = 0.99845
survprob[2] = 0.99839
survprob[3] = 0.99833
survprob[4] = 0.9983
survprob[5] = 0.99827
survprob[6] = 0.99826
survprob[7] = 0.99824
survprob[8] = 0.9982
survprob[9] = 0.99813
survprob[10] = 0.99804
survprob[11] = 0.99795
survprob[12] = 0.99785
survprob[13] = 0.99776
survprob[14] = 0.99766
survprob[15] = 0.99755
survprob[16] = 0.99743
survprob[17] = 0.9973
survprob[18] = 0.99718
survprob[19] = 0.99707
survprob[20] = 0.99696
survprob[21] = 0.99685
survprob[22] = 0.99672
survprob[23] = 0.99656
survprob[24] = 0.99635
survprob[25] = 0.9961
survprob[26] = 0.99579
survprob[27] = 0.99543
survprob[28] = 0.99504
survprob[29] = 0.99463
survprob[30] = 0.9942
survprob[31] = 0.9937
survprob[32] = 0.99311
survprob[33] = 0.99245
survprob[34] = 0.99172
survprob[35] = 0.99091
survprob[36] = 0.99005
survprob[37] = 0.98911
survprob[38] = 0.98803
survprob[39] = 0.9868
survprob[40] = 0.98545
survprob[41] = 0.98409
survprob[42] = 0.9827
survprob[43] = 0.98123
survprob[44] = 0.97961
survprob[45] = 0.97786
survprob[46] = 0.97603
survprob[47] = 0.97414
survprob[48] = 0.97207
survprob[49] = 0.9697
survprob[50] = 0.96699
survprob[51] = 0.96393
survprob[52] = 0.96055
survprob[53] = 0.9569
survprob[54] = 0.9531
survprob[55] = 0.94921
survprob[56] = 0.94508
survprob[57] = 0.94057
survprob[58] = 0.9357
survprob[59] = 0.93031
survprob[60] = 0.92424
survprob[61] = 0.91717
survprob[62] = 0.90922
survprob[63] = 0.90089
survprob[64] = 0.89282
survprob[65] = 0.88503
survprob[66] = 0.87622
survprob[67] = 0.86576
survprob[68] = 0.8544
survprob[69] = 0.8423
survprob[70] = 0.82942
survprob[71] = 0.8154
survprob[72] = 0.80002
survprob[73] = 0.78404
survprob[74] = 0.76842
survprob[75] = 0.75382
survprob[76] = 0.73996
survprob[77] = 0.72464
survprob[78] = 0.71057
survprob[79] = 0.6961
survprob[80] = 0.6809

const delta2 = delta*survprob


const tn = td-tb+1

const gr=grid.*sigma_r^0.5
const eyp=grid.*sigp_y^0.5
const eyt=grid.*sigt_y^0.5
const mu = exc+rf

const expeyp = exp.(eyp)




const galfa = linspace(0.0,1.0,nalfa)
const gret = mu + gr
const gs = range(0.05,0.5,ns) # gs now represents how much savings


# average and transitory income pieces
const f_y = zeros(nqp,tr-1)
for t = (tb+1):tr
   avg = exp(vecdot(ageprof,[1.0; t; t^2; t^3]))
   f_y[:,t-tb] = avg*exp.(eyt)
end
const ret_y= ret_fac*exp(vecdot(ageprof,[1.0; tr; tr^2; tr^3]))


# terminal/starter values
gcash =copy(gs)
c=copy(gs)
alfa=zeros(size(c))
v=utility.(c,gamma)

# policy functions
wpol = zeros(ns,80)
cpol = zeros(ns,80)
alfapol = zeros(ns,80)
vpol = zeros(ns,80)


# interpolation type (linear works just fine for me)
const inttype = Gridded(Linear()) #Gridded(Cubic(Natural()))

# start solution, retirement period first
const tt=80
for ind1 = 1:35
   t = tt-ind1+1
   age = t+tb-1
   println(string(t, " ",age))
   #interpolate cons_tp1
   itp_ctp1 = interpolate((vcat(0.0,gcash),),vcat(0.0,c),inttype)
   itp_vtp1 = interpolate((gcash,),v,inttype)
   for ind2 =  1:ns
      lowalfa2 = 1
      highalfa2 = nalfa
      if (gs[ind2] > 10) && (t < tn-1)
         lowalfa = alfa[ind2] - 0.2
         highalfa = alfa[ind2] + 0.2
         lowalfa2 = ntoi(lowalfa,galfa)
         highalfa2 = ntoi(highalfa,galfa)
      end
      nalfa_r = highalfa2 - lowalfa2 + 1
      galfa_r = galfa[lowalfa2:highalfa2]
      v1 = zeros(nalfa_r)
      for ind5 = 1:nqp
         nw = gs[ind2].*(rf .+ galfa_r .*(gret[ind5]-rf)) 
         nv = uprime.(itp_ctp1[nw .+ ret_y],gamma) .* (gret[ind5]-rf) * weig[ind5]
         v1 .+= nv
      end
      v2,pt = findmin(vec(v1).^2)
      alfa2 = galfa_r[pt]
      #println(string(t," ",gs[ind2]," ",alfa2, " ",extrema(galfa_r)))
      c1 = 0.0
      v1 = 0.0
      for ind5 = 1:nqp
         nw = gs[ind2]*(rf+ alfa2*(gret[ind5]-rf)) 
         nv = uprime(itp_ctp1[nw+ret_y],gamma) * (rf+ alfa2*(gret[ind5]-rf)) * weig[ind5]
         c1 += nv
         nvv = itp_vtp1[nw+ret_y]  * weig[ind5]
         v1 += nvv
      end
      c2 = uprimeinv(delta2[t]*c1,gamma)
      #println(string(t," ",gs[ind2]," ",alfa2, " ",c2, " ",gs[ind2] + c2))
      wpol[ind2,t] = gs[ind2] + c2
      alfapol[ind2,t] = alfa2
      cpol[ind2,t] = c2
      vpol[ind2,t] = utility(c2,gamma) + delta2[t]*v1
   end
   gcash = copy(wpol[:,t])
   alfa = copy(alfapol[:,t])
   c = copy(cpol[:,t])
   v = copy(vpol[:,t])
end

# now solve working periods
for ind1 = 1:(tt-35)
   t = 45-ind1+1
   age = t+tb-1
   println(string(t, " ",age))
   #interpolate cons_tp1
   itp_ctp1 = interpolate((vcat(0.0,gcash),),vcat(0.0,c),inttype) 
   itp_vtp1 = interpolate((gcash,),v,inttype) 
   for ind2 =  1:ns
      lowalfa2 = 1
      highalfa2 = nalfa
      if (gs[ind2] > 10) && (t < tn-1)
         lowalfa = alfa[ind2] - 0.25
         highalfa = alfa[ind2] + 0.25
         lowalfa2 = ntoi(lowalfa,galfa)
         highalfa2 = ntoi(highalfa,galfa)
      end
      nalfa_r = highalfa2 - lowalfa2 + 1
      galfa_r = galfa[lowalfa2:highalfa2]
      v1 = zeros(nalfa_r)
      @inbounds for ind5 = 1:nqp
         nw = gs[ind2].*(rf .+ galfa_r .*(gret[ind5]-rf))   
         @inbounds for ind6 = 1:nqp, ind7 = 1:nqp
            ctp1 = nw.+f_y[ind6,t]*(expeyp[ind7]+reg_coef*gret[ind5])
            nv = uprime.(itp_ctp1[ctp1],gamma) .* (gret[ind5]-rf) * (weig[ind5]*weig[ind6]*weig[ind7])
            v1 .+= nv
         end
      end
      v2,pt = findmin(vec(delta2[t]*gs[ind2]*v1).^2)
      alfa2 = galfa_r[pt]
      #println(string(t," ",gs[ind2]," ",alfa2, " ",extrema(galfa_r)))
      c1 = 0.0
      v1 = 0.0
      @inbounds for ind5 = 1:nqp
         nw = gs[ind2]*(rf+ alfa2*(gret[ind5]-rf))
         @inbounds for ind6 = 1:nqp, ind7 = 1:nqp
            nv = uprime.(itp_ctp1[nw+f_y[ind6,t]*(expeyp[ind7]+reg_coef*gret[ind5])],gamma) .* (rf+ alfa2*(gret[ind5]-rf)) * (weig[ind5]*weig[ind6]*weig[ind7])
            c1 += nv
            nvv = itp_vtp1[nw+f_y[ind6,t]*(expeyp[ind7]+reg_coef*gret[ind5])] * (weig[ind5]*weig[ind6]*weig[ind7])
            v1 += nvv
         end
      end
      c2 = uprimeinv(delta2[t]*c1,gamma)
      #println(string(t," ",gs[ind2]," ",alfa2, " ",c2, " ",gs[ind2] + c2))
      wpol[ind2,t] = gs[ind2] + c2
      alfapol[ind2,t] = alfa2
      cpol[ind2,t] = c2
      vpol[ind2,t] = utility(c2,gamma) + delta2[t]*v1
   end
   gcash = copy(wpol[:,t])
   alfa = copy(alfapol[:,t])
   c = copy(cpol[:,t])
   v = copy(vpol[:,t])
end

# write out policy and value functions to csv for inspection if you want
writecsv("testalfa_egm.csv",alfapol)
writecsv("testcons_egm.csv",cpol)
writecsv("testwealth_egm.csv",wpol)
writecsv("testvfun_egm.csv",vpol)




############################# SIMULATION
# This begins my own code (CGM did not post simualtion code)

Nhh = 10000

function simincome(Nhh,sig_t,sig_p)
	# deterministic income profile
	coefs = [-2.170042+2.700381; 0.16818; -0.0323371/10; 0.0019704/100]
	
	reprate = 0.68212
	income = zeros(80,Nhh)
	prm = zeros(80,Nhh)
	prm[1,:] = exp.(sig_p.*randn(1,Nhh))
   tb = 20
	income[1,:] = exp.(vecdot([1.0;(tb+1);(tb+1)^2;(tb+1)^3],coefs)) .* prm[1,:]' .* exp.(sig_t.*randn(1,Nhh))
	for age = (tb+2):tr
		ind = age-tb
      avg = exp.(vecdot([1.0;age;age^2;age^3],coefs))	
      prm[ind,:] = prm[ind-1,:]' .* exp.(sig_p.*randn(1,Nhh))
      income[ind,:] = avg .* prm[ind,:]' .* exp.(sig_t.*randn(1,Nhh))
   end
   income[(tr-tb+1):end,:] .= reprate .* exp.(vecdot([1.0;tr;tr^2;tr^3],coefs)) .* prm[tr-tb,:]' #.* exp.(sig_t.*randn(td-tr,Nhh))
	return income
end

income = simincome(Nhh,sigt_y^0.5,sigp_y^0.5)

cons = similar(income)
theta = similar(income)
wealth = similar(income)
wealth[1,:] = 5.0 #exp.(1.0 + randn(Nhh))

println("Simulating")
for t = 1:80 #(tb+1):td
   age = t+tb-1
   println(string(t, " ",age))
	# set up proper policy interpolations
	itp_cons = interpolate((vcat(0.0,wpol[:,t]),),vcat(0.0,cpol[:,t]),inttype)  
	itp_theta = interpolate((vcat(0.0,wpol[:,t]),),vcat(0.0,alfapol[:,t]),inttype)
	
	
	theta[t,:] = clamp.(itp_theta[wealth[t,:]],0.0,1.0)
	cons[t,:] = itp_cons[wealth[t,:]]
	
	if age+1<td
      ret = exc .+ (sigma_r^0.5 .* randn(Nhh))
		wealth[t+1,:] = income[t,:] .+ (wealth[t,:] .- cons[t,:]).*(rf .+ theta[t,:].*ret) 
	end
end


writecsv("simalfa_egm.csv",theta)
writecsv("simcons_egm.csv",cons)
writecsv("simcash_egm.csv",wealth)
writecsv("simincome_egm.csv",income)


println("Done.")

