tolerance = .0001

completed_loops=0

go = True

example = McGariConsumeType(**IdiosyncDict)

while go:
    
    example.solve()

    example.track_vars = ['aNrm','mNrm','cNrm','pLvl']
    example.initialize_sim()
    example.simulate()

    a = example.state_now['aLvl']
    AggA = np.mean(np.array(a))
    
    

    if AggA - .75 > 0 :
        
        example.Rfree = example.Rfree - .00001
        
    elif AggA - .75 < 0: 
        example.Rfree = example.Rfree + .00001
        
    else:
        break
    
    print(example.Rfree)
    
    distance = abs(AggA - .75) 
    
    completed_loops += 1
    go = distance >= tolerance and completed_loops < 100
        
    
a = example.state_now['aLvl']
c = (example.state_now['mNrm'] - example.state_now['aNrm'] ) * example.state_now['pLvl']

AggA = np.mean(np.array(a))
AggC = np.mean(np.array(c))
print(AggA)
print(AggC)