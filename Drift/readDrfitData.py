import numpy as np
import glob, os
from driftDataPreprocessing import sequenceNormalize, dataSplitShuffle

class DataFormat(object):

    @staticmethod
    def readDriftData(dir, pattern):
        driftData = dict()
        for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))
            driftData[title] = np.loadtxt(pathAndFilename, delimiter=',', skiprows=1)[:450, 1:]
            driftData[title] = np.insert(driftData[title], 3, int(title), axis=1)

        return np.stack(driftData.values())

    def load_data(self):
        # Load the data
        driftData1 = DataFormat.readDriftData('./Data/Drift/NoDrift', '*.*')
        driftData2 = DataFormat.readDriftData('./Data/Drift/Drift', '*.*')

        driftData = np.vstack([driftData1, driftData2])

        # Define Data Dimensions
        driftDataAxis0 = driftData.shape[0]
        driftDataAxis1 = driftData.shape[1]
        driftDataAxis2 = driftData.shape[2]

        # Flatten Data into 2d for scaling them
        flattenData = driftData.reshape(driftDataAxis0 * driftDataAxis1, driftDataAxis2)

        # Save the scaling algorithm for future use
        scaler = sequenceNormalize(flattenData)

        # Scale Data
        flattenData = scaler.transform(flattenData)

        # Reshape Data back into 3d (samples, time, features)
        driftDataReshape = flattenData.reshape(driftDataAxis0, driftDataAxis1, driftDataAxis2)

        y_train1 = [(1, 0) for x in range(100*driftDataAxis1)]
        y_train2 = [(0, 1) for x in range(34*driftDataAxis1)]
        y_train = y_train1+y_train2
        y_train = np.array(y_train)
        y_train = y_train.reshape(driftDataAxis0, driftDataAxis1, 2)
        # print(y_train.shape)

        # Split and shuffle the data into training and testing sets
        X_train, X_test, y_train, y_test = dataSplitShuffle(driftDataReshape, y_train)

        return ((X_train, X_test, y_train, y_test), (driftDataAxis0, driftDataAxis1, driftDataAxis2))
