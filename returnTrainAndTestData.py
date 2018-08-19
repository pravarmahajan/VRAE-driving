import numpy as np

def shuffle_in_union(data, keys):
    rng_state = np.random.get_state()
    np.random.shuffle(keys)
    np.random.set_state(rng_state)
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    return data, keys

def load_extended_data(filename):
    return np.load(filename)


def get_risk_labels():
    risk_path = "/users/PAS0536/osu9965/Telematics/Trajectories/Selected Drivers/DriverIdToRisk_old.csv"
    with open(risk_path, 'r') as f:
        lines = f.readlines()
    risk_labels = dict()

    for line in lines[1:]:
        driver, risk = line.strip().split(',')
        risk = float(risk)
        risk_labels[driver] = risk
    
    return risk_labels

def returnTrainAndTestData(args, suffix, normalization):

    extended_data = "extended_data3"
    import pickle as cPickle;
    if normalization == 1:
        norm_suffix = ""
    else:
        norm_suffix = "_" + str(normalization)
    matrices = load_extended_data('{}/extendedSample_{}_{}{}{}.npy'.format(extended_data, args[0], args[1], norm_suffix, suffix))
    keys = cPickle.load(open('{}/extendedSample_{}_{}_keys{}{}.pkl'.format(extended_data, args[0], args[1], norm_suffix, suffix), 'rb'))        
    FEATURES = matrices.shape[-1]
    
    # load Train, Dev, Test assignment
    driverToTrajectoryToSet = cPickle.load(open('{}/driverToTrajectoryToSet_{}_{}.pkl'.format(extended_data, args[0], args[1]), 'rb'))
    #risk_labels = get_risk_labels()
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    
    curTraj = ''
    assign = ''
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr
        m = matrices[idx][1:129,]
        #print (d, t, idx, m.shape)    
        if t != curTraj:
            curTraj = t
            #r = random.random()                    
            assign = (driverToTrajectoryToSet[d])[t]
        if m.shape[0] < 128:
          continue  
        #m = np.transpose(m) #need this step and the next one for CNN, but not for RNN
        #m = np.reshape(m, FEATURES*128)
        #if r < .8:
        if assign == 'train':
          train_data.append(m)
          train_labels.append(dr)
        #elif r < .9:
        elif assign == 'dev':
            dev_data.append(m)
            dev_labels.append(dr)
        else:
          test_data.append(m)
          test_labels.append(dr)        

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data     = np.asarray(dev_data, dtype="float32")
    dev_labels   = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")

    train_data, train_labels = shuffle_in_union(train_data, train_labels)   #Does shuffling do any help ==> it does a great help!!
  
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, len(driverIds)+1, FEATURES
