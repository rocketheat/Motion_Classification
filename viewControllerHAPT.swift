class ViewController: UIViewController {

    @IBOutlet weak var testing: UITextField!
    var motionManager: CMMotionManager! = nil

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        motionManager = CMMotionManager()

        motionManager.accelerometerUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)
        motionManager.gyroUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)

        motionManager.startAccelerometerUpdates(to: .main) { accelerometerData, error in
            guard let accelerometerData = accelerometerData else { return }

            self.motionManager.startGyroUpdates(to: .main) { gyroData, error in
                guard let gyroData = gyroData else {return}

                // Add the current data sample to the data array
                self.addAccelSampleToDataArray(accelSample: accelerometerData, gyroSample: gyroData)
            }
        }
    }

    struct ModelConstants {
        static let numOfFeatures = 6
        static let predictionWindowSize = 50
        static let sensorsUpdateInterval = 1.0 / 50.0
        static let hiddenInLength = 200
        static let hiddenCellInLength = 200
    }

    let activityClassificationModel = MyActivityClassifier()

    var currentIndexInPredictionWindow = 0
    let predictionWindowDataArray = try? MLMultiArray(shape: [1 , ModelConstants.predictionWindowSize , ModelConstants.numOfFeatures] as [NSNumber], dataType: MLMultiArrayDataType.double)
    var lastHiddenOutput = try? MLMultiArray(shape:[ModelConstants.hiddenInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    var lastHiddenCellOutput = try? MLMultiArray(shape:[ModelConstants.hiddenCellInLength as NSNumber], dataType: MLMultiArrayDataType.double)


    func addAccelSampleToDataArray (accelSample: CMAccelerometerData, gyroSample: CMGyroData) {
        // Add the current accelerometer reading to the data array
        guard let dataArray = predictionWindowDataArray else { return }
        dataArray[[0 , currentIndexInPredictionWindow ,0] as [NSNumber]] = accelSample.acceleration.x as NSNumber
        dataArray[[0 , currentIndexInPredictionWindow ,1] as [NSNumber]] = accelSample.acceleration.y as NSNumber
        dataArray[[0 , currentIndexInPredictionWindow ,2] as [NSNumber]] = accelSample.acceleration.z as NSNumber
        dataArray[[0 , currentIndexInPredictionWindow ,3] as [NSNumber]] = gyroSample.rotationRate.x as NSNumber
        dataArray[[0 , currentIndexInPredictionWindow ,4] as [NSNumber]] = gyroSample.rotationRate.y as NSNumber
        dataArray[[0 , currentIndexInPredictionWindow ,5] as [NSNumber]] = gyroSample.rotationRate.z as NSNumber

        // Update the index in the prediction window data array
        currentIndexInPredictionWindow += 1

        // If the data array is full, call the prediction method to get a new model prediction.
        // We assume here for simplicity that the Gyro data was added to the data array as well.
        if (currentIndexInPredictionWindow == ModelConstants.predictionWindowSize) {
            let predictedActivity = performModelPrediction() ?? "N/A"

            // Use the predicted activity here
            // ...
            print(predictedActivity)

            self.testing.text = String(predictedActivity)

            // Start a new prediction window
            currentIndexInPredictionWindow = 0
        }
    }

    func performModelPrediction () -> String? {
        guard let dataArray = predictionWindowDataArray else { return "Error!"}

        // Perform model prediction
        let modelPrediction = try? activityClassificationModel.prediction(features: dataArray, hiddenIn: lastHiddenOutput, cellIn: lastHiddenCellOutput)

        // Update the state vectors
        lastHiddenOutput = modelPrediction?.hiddenOut
        lastHiddenCellOutput = modelPrediction?.cellOut

        // Return the predicted activity - the activity with the highest probability
        return modelPrediction?.activity
    }

}
