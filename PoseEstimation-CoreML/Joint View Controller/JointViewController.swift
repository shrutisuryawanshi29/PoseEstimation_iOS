//
//  ViewController.swift
//  PoseEstimation-CoreML
//
//  Created by GwakDoyoung on 05/07/2018.
//  Copyright © 2018 tucan9389. All rights reserved.
//

import UIKit
import Vision
import CoreMedia
import os.signpost
import onnxruntime_objc
import CoreML

protocol DelegateForPAM {
    func sendTheInference(inferenceTime: Double)
}

class JointViewController: UIViewController {
    
    let refreshLog = OSLog(subsystem: "com.tucan9389.PoseEstimation-CoreML", category: "InferenceOperations")
    
    public typealias DetectObjectsCompletion = ([PredictedPoint?]?, Error?) -> Void
    
    // MARK: - UI Properties
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var jointView: DrawingJointView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    @IBOutlet weak var inferenceLabel: UILabel!
    @IBOutlet weak var etimeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    @IBOutlet weak var btnSelectModel: UIButton!
    
    //MARK: - Variable declaration
    private lazy var menu = UIMenu(title: "Select Model", children: elements)
    
    private lazy var first = UIAction(title: "model_cpm", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .MODEL_CPM)
    }
    
    private lazy var second = UIAction(title: "yolov8n-pose", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8N_POSE)
    }
    
    private lazy var third = UIAction(title: "yolov8s-pose", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8S_POSE)
    }
    
    private lazy var four = UIAction(title: "yolov8m-pose", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8M_POSE)
    }
    
    private lazy var five = UIAction(title: "yolov8l-pose", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8L_POSE)
    }
    
    private lazy var six = UIAction(title: "yolov8x-pose", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8X_POSE)
    }
    
    private lazy var seven = UIAction(title: "yolov8x-pose-p6", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLOV8X_POSE_P6)
    }
    
    private lazy var nine = UIAction(title: "yolo_nas_pose_l", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .YOLO_NAS_POSE_L)
    }
    
    private lazy var eight = UIAction(title: "PAM", attributes: [], state: .off) { action in
        self.selectWhichModel(modelType: .PAM)
    }
    
    private lazy var elements: [UIAction] = [first, second, third, four, five, six, seven, eight, nine]
    
    var currModelType: ModelType = .MODEL_CPM
    var inferenceTimeGlobal:Int = 0
    
    // MARK: - Performance Measurement Property
    private let 👨‍🔧 = 📏()
    var isInferencing = false
    
    // MARK: - AV Property
    var videoCapture: VideoCapture!
    
    // MARK: - ML Properties
    // Core ML model
    typealias EstimationModel = model_cpm
    
    // Preprocess and Inference
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    // Postprocess
    var postProcessor: HeatmapPostProcessor = HeatmapPostProcessor()
    var mvfilters: [MovingAverageFilter] = []
    
    // Inference Result Data
    private var tableData: [PredictedPoint?] = []
    
    // Handle all model and data preprocessing and run inference
    private var modelHandler: ModelHandler? = ModelHandler(
        modelFileInfo: (name: "rtmpose-m-27c0e6", extension: "ort"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        btnSelectModel.showsMenuAsPrimaryAction = true
        btnSelectModel.menu = menu
        
        // setup the model PoseEstimation
        setUpModel()
        
        //setup ONNX model
        //setupOnnxModel()
        
        //setup model for YOLO
        //setupPoseEsti(modelType: .YOLOV8S_POSE)
        
        // setup camera
        setUpCamera()
        
        // setup tableview datasource on bottom
        labelsTableView.dataSource = self
        
        // setup delegate for performance measurement
        👨‍🔧.delegate = self
        
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    //MARK: - User defined methods
    
    //Function to setup dropdowns for models
    func selectWhichModel(modelType: ModelType) {
        self.currModelType = modelType
        modelHandler?.delegate = self
        switch (modelType) {
            
        case .MODEL_CPM:
            btnSelectModel.setTitle("model_cpm", for: .normal)
            setUpModel()
            
        case .YOLOV8N_POSE:
            btnSelectModel.setTitle("yolov8n-pose", for: .normal)
            setupPoseEsti(modelType: .YOLOV8N_POSE)
            
        case .YOLOV8S_POSE:
            btnSelectModel.setTitle("yolov8s-pose", for: .normal)
            setupPoseEsti(modelType: .YOLOV8S_POSE)
            
        case .YOLOV8M_POSE:
            btnSelectModel.setTitle("yolov8m-pose", for: .normal)
            setupPoseEsti(modelType: .YOLOV8M_POSE)
            
        case .YOLOV8L_POSE:
            btnSelectModel.setTitle("yolov8l-pose", for: .normal)
            setupPoseEsti(modelType: .YOLOV8L_POSE)
            
        case .YOLOV8X_POSE:
            btnSelectModel.setTitle("yolov8x-pose", for: .normal)
            setupPoseEsti(modelType: .YOLOV8X_POSE)
            
        case .YOLOV8X_POSE_P6:
            btnSelectModel.setTitle("yolov8x-pose-p6", for: .normal)
            setupPoseEsti(modelType: .YOLOV8X_POSE_P6)
            
        case .PAM:
            btnSelectModel.setTitle("PAM", for: .normal)
            setupPoseEsti(modelType: .YOLOV8N_POSE)
        
        case .YOLO_NAS_POSE_L:
            btnSelectModel.setTitle("yolo_nas_pose_l", for: .normal)
            setupPoseEsti(modelType: .YOLO_NAS_POSE_L)
        }
    }
    
    @IBAction func btnSelectModelClick(_ sender: Any) {
    }
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: EstimationModel().model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("cannot load the ml model")
        }
    }
    
    func setupPoseEsti(modelType: ModelType) {
        guard let modelURL = Bundle.main.url(forResource: modelType.rawValue, withExtension: "mlmodelc") else {
            return
        }
        
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidCompleteForYolov8)
            request?.imageCropAndScaleOption = .scaleFill
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
    }
    
    // MARK: - SetUp Video
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .vga640x480) { success in
            
            if success {
                if let previewLayer = self.videoCapture.previewLayer {
                    DispatchQueue.main.async {
                        self.videoPreview.layer.addSublayer(previewLayer)
                        self.resizePreviewLayer()
                    }
                }
                
                // start video preview when setup is done
                self.videoCapture.start()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        resizePreviewLayer()
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
}

// MARK: - VideoCaptureDelegate
extension JointViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        // the captured image from camera is contained on pixelBuffer
        if !isInferencing {
            
            isInferencing = true
            
            // start of measure
            self.👨‍🔧.🎬👏()
            
            // predict!
            self.predictUsingVision(pixelBuffer: pixelBuffer)
        }
    }
}

extension JointViewController {
    // MARK: - Inferencing
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        
        if #available(iOS 12.0, *) {
            os_signpost(.begin, log: refreshLog, name: "PoseEstimation")
        }
        try? handler.perform([request])
    }
    
    // MARK: - Postprocessing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if #available(iOS 12.0, *) {
            os_signpost(.event, log: refreshLog, name: "PoseEstimation")
        }
        self.👨‍🔧.🏷(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let heatmaps = observations.first?.featureValue.multiArrayValue {

            /* =================================================================== */
            /* ========================= post-processing ========================= */

            /* ------------------ convert heatmap to point array ----------------- */
            var predictedPoints = postProcessor.convertToPredictedPoints(from: heatmaps)

            /* --------------------- moving average filter ----------------------- */
            if predictedPoints.count != mvfilters.count {
                mvfilters = predictedPoints.map { _ in MovingAverageFilter(limit: 3) }
            }
            for (predictedPoint, filter) in zip(predictedPoints, mvfilters) {
                filter.add(element: predictedPoint)
            }
            predictedPoints = mvfilters.map { $0.averagedValue() }
            /* =================================================================== */

            /* =================================================================== */
            /* ======================= display the results ======================= */
            DispatchQueue.main.sync {
                // draw line
                self.jointView.bodyPoints = predictedPoints

                // show key points description
                self.showKeypointsDescription(with: predictedPoints)

                // end of measure
                self.👨‍🔧.🎬🤚()
                self.isInferencing = false
                
                if #available(iOS 12.0, *) {
                    os_signpost(.end, log: refreshLog, name: "PoseEstimation")
                }
            }
            /* =================================================================== */
        } else {
            // end of measure
            self.👨‍🔧.🎬🤚()
            self.isInferencing = false
            
            if #available(iOS 12.0, *) {
                os_signpost(.end, log: refreshLog, name: "PoseEstimation")
            }
        }
    }
    
    func showKeypointsDescription(with n_kpoints: [PredictedPoint?]) {
        self.tableData = n_kpoints
        self.labelsTableView.reloadData()
    }
}

// MARK: - UITableView Data Source
extension JointViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tableData.count// > 0 ? 1 : 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell: UITableViewCell = tableView.dequeueReusableCell(withIdentifier: "LabelCell", for: indexPath)
        cell.textLabel?.text = PoseEstimationForMobileConstant.pointLabels[indexPath.row]
        if let body_point = tableData[indexPath.row] {
            let pointText: String = "\(String(format: "%.3f", body_point.maxPoint.x)), \(String(format: "%.3f", body_point.maxPoint.y))"
            cell.detailTextLabel?.text = "(\(pointText)), [\(String(format: "%.3f", body_point.maxConfidence))]"
        } else {
            cell.detailTextLabel?.text = "N/A"
        }
        return cell
    }
}

// MARK: - 📏(Performance Measurement) Delegate
extension JointViewController: 📏Delegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        if currModelType == .PAM {
            self.inferenceTimeGlobal = Int(inferenceTime*1000.0)
            self.etimeLabel.text = "execution: \(Int(executionTime*1000.0)) ms"
            self.fpsLabel.text = "fps: \(fps)"
        }
        else {
            self.inferenceLabel.text = "inference: \(Int(inferenceTime*1000.0)) ms"
            self.etimeLabel.text = "execution: \(Int(executionTime*1000.0)) ms"
            self.fpsLabel.text = "fps: \(fps)"
        }
    }
}

//MARK: YOLOv8 Models
extension JointViewController {
    
}

//MARK: ONNX Models
extension JointViewController {

    func setupOnnxModel() {
        print(modelHandler)
    }
    
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
//        guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs
//        else { return }
//        previousInferenceTimeMs = currentTimeMs
        
        
        // Display results by the `InferenceViewController`.
        DispatchQueue.main.async {
            let resolution = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                                    height: CVPixelBufferGetHeight(pixelBuffer))
//            self.inferenceViewController?.resolution = resolution
//            
//            var inferenceTime: Double = 0
//            if let resultInferenceTime = self.result?.processTimeMs {
//                inferenceTime = resultInferenceTime
//            }
//            self.inferenceViewController?.inferenceTime = inferenceTime
//            self.inferenceViewController?.tableView.reloadData()
//            
//            // Draw bounding boxes and compute the inference score
//            self.drawBoundingBoxesAndCalculate(onInferences: displayResult.inferences,
//                                               withImageSize: CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
//                                                                     height: CVPixelBufferGetHeight(pixelBuffer)))
        }
    }
    
    // MARK: - Postprocessing
    func visionRequestDidCompleteForYolov8(request: VNRequest, error: Error?) {
        if #available(iOS 12.0, *) {
            os_signpost(.event, log: refreshLog, name: "PoseEstimation")
        }
        self.👨‍🔧.🏷(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let heatmaps = observations.first?.featureValue.multiArrayValue {
            
            DispatchQueue.main.sync {
                self.jointView.bodyPoints = []
                // draw line
                self.jointView.bodyPointsForYOLO = (self.modelHandler?.parsePoseYOLOv8(results: observations, modelType: currModelType))!
                
                // end of measure
                self.👨‍🔧.🎬🤚()
                self.isInferencing = false
                
                if #available(iOS 12.0, *) {
                    os_signpost(.end, log: refreshLog, name: "PoseEstimation")
                }
            }
        }
        else {
            // end of measure
            self.👨‍🔧.🎬🤚()
            self.isInferencing = false
            
            if #available(iOS 12.0, *) {
                os_signpost(.end, log: refreshLog, name: "PoseEstimation")
            }
        }
    }
    
    
}

extension JointViewController: DelegateForPAM {
    func sendTheInference(inferenceTime: Double) {
        print("from the delegate \(inferenceTime)")
        
        var strInference = "Inference for YOLO \(self.inferenceTimeGlobal) ms\nInference for PAM \(Int(inferenceTime)) ms\nInference for YOLO and PAM: \(Int(inferenceTime) + self.inferenceTimeGlobal) ms"
        
        
        self.inferenceLabel.text = strInference
        self.etimeLabel.text = "execution: xxx ms"
        self.fpsLabel.text = "fps: xxx"
    }
    
    
}
