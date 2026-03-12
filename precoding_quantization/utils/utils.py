import os
import numpy as np
import json
import re

def create_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def C2R(HC):
    """
    convert complex channel matrix into the re and im parts
    :param H: K x M complex channel matrix
    :return: H: K x M x 2 channel matrix with a real and an imaginary 'color channel'
    """
    Hre = HC.real
    Him = HC.imag
    HR = np.stack((Hre, Him), axis=-1)
    return HR

def rayleigh_channel_MU(M, K):
    """
    :param M: number of antennas
    :param K: number of users
    :return: H (MxK) complex gaussian distributed channel: ~CN(0,1) = ~ N(0,1/2) + N(0,1/2) * j
    """
    variance = 1/2
    stdev = np.sqrt(variance)
    H = np.zeros((M, K), dtype=complex)

    for k in range(K):
        H[:, k] = np.random.normal(0, stdev, M) + 1j * np.random.normal(0, stdev, M)

    return H

def getSymbols(Ndata=5000, p = 1):
    """
    :param Ndata: number of symbols to generate
    :param p: signal variance
    :return: Ndata symbols sampled from a complex gaussian with variance p
    Note that this generates a variable which is drawn from a complex gaussian distribution with variance p
    which is equivalent to a + bj with a and b sampled from a gaussian distriobution with variance p/2.
    Here we first sample a and b from a gaussian with mean 0 and variance 1, by multiplying with sqrt(p)/sqrt(2)
    we obtain variance p/2 for both a and b, given that var(constant * X) = constant^2 var(X)
    """
    s = np.sqrt(p) / np.sqrt(2) * (np.random.randn(Ndata) + 1j* np.random.randn(Ndata))
    return s

def getSymbols_QPSK(Ndata=5000):

    # QPSK connstellation
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    # Generate random bit stream
    bps = 2 # bits per symbol => 2 cause QPSK
    b = np.random.randint(0, 2, (Ndata, bps))

    # Convert to symbols
    b_flipped = b[:, [1, 0]]  # intermediate flip to deal with the np.packbits function
    idx = np.packbits(b_flipped, bitorder='little', axis=-1).flatten()
    s = constellation[idx]

    return s, b

def symbols_MU(K, nrdata):
    # generate symbols => variance = 1 => E[|wk sk|^2] = Pk
    S = np.zeros((K, nrdata), dtype=complex)
    for k in range(K):
        S[k, :] = getSymbols(nrdata, p=1)
    return S

def los_channel_MU(M, K):
    H = np.zeros((M, K), dtype=complex)

    for k in range(K):
        userangle = np.random.randint(0, 180)
        theta = userangle * np.pi / 180
        H[:, k] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))
    return H

def cellfree_channel_MU(M, K, ap_positions=None, area_side_m=100.0, min_distance_m=10.0,
                        pathloss_exponent=3.7, shadowing_std_db=8.0, pathloss_at_1m_db=-30.0):
    """
    Generate one cell-free channel realization with large-scale and small-scale fading.

    :param M: number of AP antennas (or AP-side transmit dimensions)
    :param K: number of users
    :param ap_positions: optional AP coordinates, shape (M, 2). If None, sampled uniformly.
    :param area_side_m: side length of square deployment area in meters
    :param min_distance_m: minimum AP-UE distance to avoid singular path loss
    :param pathloss_exponent: path loss exponent
    :param shadowing_std_db: standard deviation of log-normal shadowing in dB
    :param pathloss_at_1m_db: path loss reference at 1 meter in dB
    :return: (H, ap_positions)
    """
    if ap_positions is None:
        ap_positions = np.random.uniform(0.0, area_side_m, size=(M, 2))

    ue_positions = np.random.uniform(0.0, area_side_m, size=(K, 2))

    # Distances between APs and UEs: M x K
    delta = ap_positions[:, np.newaxis, :] - ue_positions[np.newaxis, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    distances = np.maximum(distances, min_distance_m)

    # Large-scale fading beta_{m,k}: path loss + log-normal shadowing
    pathloss_db = pathloss_at_1m_db - 10.0 * pathloss_exponent * np.log10(distances)
    shadowing_db = np.random.normal(0.0, shadowing_std_db, size=(M, K))
    beta = 10 ** ((pathloss_db + shadowing_db) / 10.0)

    # Small-scale fading g_{m,k} ~ CN(0, 1)
    g = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2.0)

    # Cell-free channel: h_{m,k} = sqrt(beta_{m,k}) * g_{m,k}
    H = np.sqrt(beta) * g
    return H, ap_positions

class NumpyEncoder(json.JSONEncoder):
    #json incoder fur numpy arrays
    #to undo  np.asarray(json_load["a"])
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def logparams(path, params, output_dir=None):
    copyparams = dict(params)
    if 'Bs' in params.keys():
        np.save(os.path.join(output_dir, 'Bs.npy'), copyparams['Bs'])
        del copyparams['Bs']
    f = open(path, "w")
    f.write(json.dumps(copyparams, cls=NumpyEncoder))
    f.close()

def logmodel(path, model):
    summary = str(model.to_json())
    f = open(path, 'w')
    f.write(json.dumps(summary))
    f.close

def get_data(M, K, Ntr, Nval, Nte, datafolder, channelmodel='iid'):

    #get all dataset names in the datafolder
    all_datasets = [os.path.basename(f.path) for f in os.scandir(datafolder) if f.is_dir()]

    #loop over all existing datasets
    dataexists = False
    for dataset_name in all_datasets:
        #extract dataset information from foldername
        Mdataset = int(re.findall(str(re.escape('M_')) +"(.*)" + str(re.escape('_K')) , dataset_name)[0])
        Kdataset = int(re.findall(str(re.escape('K_')) +"(.*)" + str(re.escape('_Ntr')) , dataset_name)[0])
        Ntr_dataset = int(re.findall(str(re.escape('Ntr_')) +"(.*)" + str(re.escape('_Nval')) , dataset_name)[0])
        Nval_dataset = int(re.findall(str(re.escape('Nval_')) +"(.*)" + str(re.escape('_Nte')) , dataset_name)[0])
        Nte_dataset = int(re.findall(str(re.escape('Nte_')) +"(.*)" + str(re.escape('_')) , dataset_name)[0])

        #check if the desired dataset exists
        if M == Mdataset and K == Kdataset and Ntr <= Ntr_dataset and Nval <= Nval_dataset and Nte <= Nte_dataset and channelmodel in dataset_name:
            Htrain = np.load(os.path.join(datafolder, dataset_name, 'Htrain.npy'))
            Htest = np.load(os.path.join(datafolder, dataset_name, 'Htest.npy'))
            Hval = np.load(os.path.join(datafolder, dataset_name, 'Hval.npy'))

            #only take the amount of data you need
            Htrain = Htrain[0:Ntr, :, :]
            Htest = Htest[0:Nte, :, :]
            Hval = Hval[0:Nval, :, :]
            dataexists = True
            print('Data has been loaded')
            break #exit the loop if we found the correct dataset

    #generate the dataset if it doesn't exist yet
    if dataexists == False:
        if channelmodel == 'iid':
            H = np.zeros((Ntr + Nval + Nte, M, K), dtype=complex)
            for i in range(Ntr + Nte + Nval):
                H[i, :, :] = rayleigh_channel_MU(M, K)

            # generate test and train set
            Htrain = H[0:Ntr, :, :]
            Hval = H[Ntr:Nval + Ntr, :, :]
            Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

            #save the data
            output_dir = os.path.join(datafolder, f'M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_iid')
            create_folder(output_dir)
            np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
            np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
            np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
            print('Data has been generated')

        elif channelmodel == 'los':
            H = np.zeros((Ntr + Nval + Nte, M, K), dtype=complex)
            for i in range(Ntr + Nte + Nval):
                H[i, :, :] = los_channel_MU(M, K)

            # generate test and train set
            Htrain = H[0:Ntr, :, :]
            Hval = H[Ntr:Nval + Ntr, :, :]
            Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

            #save the data
            output_dir = os.path.join(datafolder, f'M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_los')
            create_folder(output_dir)
            np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
            np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
            np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
            print('Data has been generated')

        else:
            print(f'{channelmodel} channel model not implemented')
    return Htrain, Hval, Htest

def load_params(path):
    f = open(path)
    params = json.load(f)
    return params