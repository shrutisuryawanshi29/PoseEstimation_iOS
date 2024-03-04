//
//  ThreeDPAMViewController.swift
//  PoseEstimation-CoreML
//
//  Created by Shruti Suryawanshi on 10/16/23.
//  Copyright © 2023 tucan9389. All rights reserved.
//

import UIKit
import Vision
import CoreMedia
import os.signpost
import onnxruntime_objc
import CoreML
import simd
import TFLiteSwift_Vision
import SceneKit

class ThreeDPAMViewController: UIViewController {
    
    let refreshLog = OSLog(subsystem: "com.tucan9389.PoseEstimation-CoreML", category: "InferenceOperations")
    
    public typealias DetectObjectsCompletion = ([PredictedPoint?]?, Error?) -> Void
    
    // MARK: - UI Properties
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var jointView: Pose3DSceneView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    @IBOutlet weak var inferenceLabel: UILabel!
    @IBOutlet weak var etimeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    @IBOutlet weak var mySceneView: SCNView!
    
    var currModelType: ModelType = .PAM
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
        modelFileInfo: (name: "PAM", extension: "onnx"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    var humanBodyRenderer = HumanBodySkeletonRenderer()
    
    // MARK: - ML Property
    
    var humanKeypoints: HumanKeypoints? {
        didSet {
            DispatchQueue.main.async {
                self.jointView?.humanKeypoints = self.humanKeypoints
            }
        }
    }
    var humanType: PostprocessOptions.HumanType = .singlePerson
    var postprocessOptions: PostprocessOptions {
        return PostprocessOptions(partThreshold: 0.5, // not use in 3D pose estimation
                                  bodyPart: nil,
                                  humanType: humanType)
    }
    
    struct PostprocessOptions {
        let partThreshold: Float?
        let bodyPart: Int?
        let humanType: HumanType
        
        enum HumanType {
            case singlePerson
            case multiPerson(pairThreshold: Float?, nmsFilterSize: Int, maxHumanNumber: Int?)
        }
    }
    
    
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup the model PoseEstimation
        setupPoseEsti(modelType: .YOLOV8N_POSE)
        
        // setup camera
        setUpCamera()
        
        //setupScene()
        
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
    
    func setupScene(coordinates: [[[Float32]]]) {
        //        scnView = SCNView(frame: self.view.frame)
        //        self.view.addSubview(scnView)
        
        mySceneView.allowsCameraControl = true
        mySceneView.autoenablesDefaultLighting = true
        mySceneView.scene = createSkeleton(coordinates: coordinates)
    }
    
    
    
    func createSkeleton(coordinates: [[[Float32]]]) -> SCNScene {
        var myScene = SCNScene()
        //if let humanBodyRenderer = humanBodyRenderer {
        // create and add a camera to the scene
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        myScene.rootNode.addChildNode(cameraNode)
        
        // place the camera
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 16)
        
        var nodeDict = humanBodyRenderer.createSkeletonNodes(observation: coordinates)
        print(nodeDict)
        let imagePlaneScale = humanBodyRenderer.relate3DSkeletonProportionToImagePlane(observation: nodeDict)
        humanBodyRenderer.imageNodeSize.width *= CGFloat(imagePlaneScale)
        humanBodyRenderer.imageNodeSize.height *= CGFloat(imagePlaneScale)
        
        let planeGeometry = SCNPlane(width: humanBodyRenderer.imageNodeSize.width,
                                     height: humanBodyRenderer.imageNodeSize.height)
        
        let point = humanBodyRenderer.computeOffsetOfRoot(observation: nodeDict)
        //        imageNode.simdPosition = simd_float3(x: imageNode.simdPosition.x - Float(point.x),
        //                                             y: imageNode.simdPosition.y - Float(point.y),
        //                                             z: imageNode.simdPosition.z)
        
        // Add camera representations to the scene (pyramid and new scene camera).
        //            if showCamera {
        //                myScene.rootNode.addChildNode(humanBodyRenderer.createCameraNode(observation: humanObservation!))
        //            } else {
        //myScene.rootNode.addChildNode(humanBodyRenderer.createCameraPyramidNode(observation: humanObservation!))
        //            }
        
        // Add skeleton nodes to the scene.
        let bodyAnchorNode = SCNNode()
        bodyAnchorNode.position = SCNVector3(0, 0, 0)
        bodyAnchorNode.geometry?.firstMaterial?.diffuse.contents = UIColor(ciColor: .yellow)
        myScene.rootNode.addChildNode(bodyAnchorNode)
        for jointName in nodeDict.keys {
            if let jointNode = nodeDict[jointName] {
                bodyAnchorNode.addChildNode(jointNode)
            }
        }
        
        // Give the head more spherical geometry.
        if let topHead = nodeDict[.Head], let centerHeadNode = nodeDict[.Nose], let centerShoulderNode = nodeDict[.Neck] {
            let headHight = CGFloat(topHead.position.y - centerShoulderNode.position.y)
            centerHeadNode.geometry = SCNBox(width: 0.2,
                                             height: headHight,
                                             length: 0.2,
                                             chamferRadius: 0.4)
            centerHeadNode.geometry?.firstMaterial?.diffuse.contents = UIColor(ciColor: .red)
            topHead.isHidden = true
        }
        
        let jointOrderArray: [HumanKeypointName] = [.L_wrist, .L_Elbow, .L_shoulder,
                                                    .R_wrist, .R_Elbow, .R_shoulder,
                                                    .Neck, .Torso, .R_Ankle,
                                                    .R_Knee, .R_Hip, .L_Ankle, .L_Knee, .L_Hip]
        if !nodeDict.isEmpty {
            for (joint, map) in nodeDict {
                
                connectNodeToParent(joint: joint, nodeJointDict: nodeDict)
                
            }
        }
        // }
        
        return myScene
    }
    
    func distanceBetweenPoints(pointA: SCNVector3, pointB: SCNVector3) -> Float {
        return sqrt(pow(pointB.x - pointA.x, 2) + pow(pointB.y - pointA.y, 2) + pow(pointB.z - pointA.z, 2))
    }
    
    func midpointBetweenPoints(pointA: SCNVector3, pointB: SCNVector3) -> SCNVector3 {
        return SCNVector3((pointA.x + pointB.x) / 2.0, (pointA.y + pointB.y) / 2.0, (pointA.z + pointB.z) / 2.0)
    }
    
    func calculateEulerAngles(pointA: SCNVector3, pointB: SCNVector3) -> SCNVector3 {
        let deltaX = pointB.x - pointA.x
        let deltaY = pointB.y - pointA.y
        let deltaZ = pointB.z - pointA.z
        let height = distanceBetweenPoints(pointA: pointA, pointB: pointB)
        
        let yaw = atan2(deltaY, deltaX)
        let pitch = -atan2(deltaZ, sqrt(deltaX * deltaX + deltaY * deltaY))
        return SCNVector3(pitch, yaw, 0)
    }
    
    
    // MARK: - Setup Core ML
    
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
extension ThreeDPAMViewController: VideoCaptureDelegate {
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

extension ThreeDPAMViewController {
    // MARK: - Inferencing
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        if let _ = request {
            guard let request = self.request else { fatalError() }
            // vision framework configures the input size of image following our model's input configuration automatically
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
            
            if #available(iOS 12.0, *) {
                os_signpost(.begin, log: refreshLog, name: "PoseEstimation")
            }
            try? handler.perform([request])
        }
    }
    
    // MARK: - Postprocessing
    func showKeypointsDescription(with n_kpoints: [PredictedPoint?]) {
        self.tableData = n_kpoints
        self.labelsTableView.reloadData()
    }
}

// MARK: - UITableView Data Source
extension ThreeDPAMViewController: UITableViewDataSource {
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
extension ThreeDPAMViewController: 📏Delegate {
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

//MARK: ONNX Models
extension ThreeDPAMViewController {
    
    // MARK: - Postprocessing
    func visionRequestDidCompleteForYolov8(request: VNRequest, error: Error?) {
        if #available(iOS 12.0, *) {
            os_signpost(.event, log: refreshLog, name: "PoseEstimation")
        }
        var result: Result<PoseEstimationOutput, PoseEstimationError>
        self.👨‍🔧.🏷(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let heatmaps = observations.first?.featureValue.multiArrayValue {
            
            // draw line
            var outputs = (self.modelHandler?.parsePoseYOLOv8ForPAM(results: observations, modelType: currModelType))!
            
            print("outputs \(outputs)")
            print("reshapeArray1DTo3D - \(reshapeArray1DTo3D(array1D: outputs))")
            //createSkeleton(coordinates: reshapeArray1DTo3D(array1D: outputs) ?? [[[Float32]]]())
            
            var jointCoordinates: [[[Float32]]] =
            [[[0], [-0.20], [0]], //0
             [[0.20], [-0.30], [0]], //1
             [[0.20], [-0.50], [0]], //2
             [[0.20], [-0.70], [0]], //3
             [[-0.20], [-0.30], [0]], //4
             [[-0.20], [-0.50], [0]], //5
             [[-0.20], [-0.70], [0]], //6
             [[0], [0], [0]], //7
             [[0], [0.20], [0]], //8
             [[0], [0.40], [0]], //9
             [[0], [0.50], [0]], //10
             [[-0.20], [0.20], [0]], //11
             [[-0.30], [0], [0]], //12
             [[-0.30], [-0.10], [0]], //13
             [[0.20], [0.20], [0]], //14
             [[0.30], [0], [0]], //15
             [[0.30], [-0.10], [0]]] //16
            
            if(!outputs.isEmpty) {
                setupScene(coordinates: reshapeArray1DTo3D(array1D: outputs) ?? [[[Float32]]]())
                //setupScene(coordinates: jointCoordinates)
            }
            DispatchQueue.main.sync {
                
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
    
    
    func reshapeArray1DTo3D(array1D: [Float32], depth: Int = 17, rows: Int = 3, columns: Int = 1) -> [[[Float32]]]? {
        let totalElements = depth * rows * columns
        if array1D.count != totalElements {
            print("Reshaping not possible due to mismatch in total elements")
            return nil
        }
        
        var array3D = [[[Float32]]](repeating: [[Float32]](repeating: [Float32](repeating: 0, count: columns), count: rows), count: depth)
        
        for i in 0..<totalElements {
            let d = i / (rows * columns)
            let r = (i % (rows * columns)) / columns
            let c = i % columns
            array3D[d][r][c] = array1D[i]
        }
        
        return array3D
    }
    
}

extension ThreeDPAMViewController: DelegateForPAM {
    func sendTheInference(inferenceTime: Double) {
        print("from the delegate \(inferenceTime)")
        
        var strInference = "Inference for YOLO \(self.inferenceTimeGlobal) ms\nInference for PAM \(Int(inferenceTime)) ms\nInference for YOLO and PAM: \(Int(inferenceTime) + self.inferenceTimeGlobal) ms"
        
        
        self.inferenceLabel.text = strInference
        self.etimeLabel.text = "execution: xxx ms"
        self.fpsLabel.text = "fps: xxx"
    }
    
    
}


private extension CGRect {
    func scaled(to scalingRatio: CGFloat) -> CGRect {
        return CGRect(x: origin.x * scalingRatio, y: origin.y * scalingRatio,
                      width: width * scalingRatio, height: height * scalingRatio)
    }
}

private extension Array where Element == PoseEstimationOutput.Human3D.Line3D {
    func matchVector(with capturedLines: [PoseEstimationOutput.Human3D.Line3D]) -> CGFloat {
        let cosineSimilaries = zip(capturedLines, self).map { (capturedLine, predictedLine) -> CGFloat in
            let v1 = capturedLine.to - capturedLine.from
            let v2 = predictedLine.to - predictedLine.from
            return v1.product(rhs: v2) / (v1.distance * v2.distance)
        }
        let averageSilirarity = cosineSimilaries.reduce(0.0) { $0 + $1 } / CGFloat(cosineSimilaries.count)
        
        return averageSilirarity
    }
    
    
}

extension PoseEstimationOutput.Human3D {
    func adjustLines() -> [PoseEstimationOutput.Human3D.Line3D] {
        guard let index1 = baselineKeypointIndexes?.0, let index2 = baselineKeypointIndexes?.1 else { return [] }
        guard let kp1 = keypoints[index1], let kp2 = keypoints[index2] else { return [] }
        
        let kp1_f = kp1.position.simdVector
        let kp2_f = kp2.position.simdVector
        let kp_m_f = (kp1_f + kp2_f) / 2.0
        
        let (theta1, theta2) = getThetas(kp_f: kp1_f, kp_m_f: kp_m_f)
        
        return lines.map { line -> (from: Keypoint3D, to: Keypoint3D) in
            let from = line.from.adjustKeypoint(theta1: theta1, theta2: theta2, kp_m_f: kp_m_f)
            let to = line.to.adjustKeypoint(theta1: theta1, theta2: theta2, kp_m_f: kp_m_f)
            return (from: from, to: to)
        }
    }
    
    func adjustKeypoints() -> [Keypoint3D?] {
        guard let index1 = baselineKeypointIndexes?.0, let index2 = baselineKeypointIndexes?.1 else { return [] }
        guard let kp1 = keypoints[index1], let kp2 = keypoints[index2] else { return [] }
        
        let kp1_f = kp1.position.simdVector
        let kp2_f = kp2.position.simdVector
        let kp_m_f = (kp1_f + kp2_f) / 2.0
        
        let (theta1, theta2) = getThetas(kp_f: kp1_f, kp_m_f: kp_m_f)
        
        return keypoints.map { keypoint in
            return keypoint?.adjustKeypoint(theta1: theta1, theta2: theta2, kp_m_f: kp_m_f)
        }
    }
    
    func getThetas(kp_f: simd_float3, kp_m_f: simd_float3) -> (theta1: Float, theta2: Float) {
        let moved_kp_f = kp_f - kp_m_f
        let theta1: Float = atan(moved_kp_f.y / moved_kp_f.x) // radian
        let roated_kp_f = moved_kp_f.rotate(angle: -theta1, axis: .zAxis)
        let theta2: Float = atan(roated_kp_f.z / roated_kp_f.x) // radian
        return (theta1, theta2)
    }
}

extension Keypoint3D {
    func adjustKeypoint(theta1: Float, theta2: Float, kp_m_f: simd_float3) -> Keypoint3D {
        let kp_f = position.simdVector
        let moved_kp_f = kp_f - kp_m_f
        let roated_kp_f = moved_kp_f.rotate(angle: -theta1, axis: .zAxis).rotate(angle: -theta2, axis: .yAxis)
        let middlex_kp_m_f = simd_float3(x: 0.5, y: kp_m_f.y, z: kp_m_f.z)
        let movebacked_kp_f = roated_kp_f + middlex_kp_m_f
        return movebacked_kp_f.keypoint
    }
}

extension simd_float3 {
    var keypoint: Keypoint3D {
        return Keypoint3D(x: CGFloat(x), y: CGFloat(y), z: CGFloat(z))
    }
}


extension simd_float3 {
    enum RotateAxis {
        case xAxis
        case yAxis
        case zAxis
    }
    
    func rotate(angle: Float, axis: RotateAxis) -> simd_float3 {
        let rows: [simd_float3]
        switch axis {
        case .xAxis:
            rows = [
                simd_float3(1,          0,           0),
                simd_float3(0, cos(angle), -sin(angle)),
                simd_float3(0, sin(angle),  cos(angle)),
            ]
        case .yAxis:
            rows = [
                simd_float3(cos(angle), 0, -sin(angle)),
                simd_float3(0,          1,           0),
                simd_float3(sin(angle), 0,  cos(angle)),
            ]
        case .zAxis:
            rows = [
                simd_float3(cos(angle), -sin(angle), 0),
                simd_float3(sin(angle),  cos(angle), 0),
                simd_float3(0,           0,          1),
            ]
        }
        
        return float3x3(rows: rows) * self
    }
    
    static func + (lhs: simd_float3, rhs: simd_float3) -> simd_float3 {
        return simd_float3(x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z)
    }
    
    static func - (lhs: simd_float3, rhs: simd_float3) -> simd_float3 {
        return simd_float3(x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z)
    }
    
    static func / (lhs: simd_float3, rhs: Float) -> simd_float3 {
        return simd_float3(x: lhs.x / rhs, y: lhs.y / rhs, z: lhs.z / rhs)
    }
}


extension ThreeDPAMViewController {
    // MARK: - Redraws the skeleton upon model change.
    func connectNodeToParent(joint: HumanKeypointName,
                             nodeJointDict: [HumanKeypointName: SCNNode]) {
        updateLineNode(node: nodeJointDict[joint]!,
                       joint: joint,
                       fromPoint: nodeJointDict[joint]!.simdPosition,
                       toPoint: nodeJointDict[parentJointNameForJointName(jointName: joint)]!.simdPosition)
    }
    
    func updateLineNode(node: SCNNode,
                        joint: HumanKeypointName,
                        fromPoint: simd_float3,
                        toPoint: simd_float3,
                        originalCubeWidth: Float = 0.05) {
        // Determine the distance between the child and parent nodes.
        let length = max(simd_length(toPoint - fromPoint), 1E-5)
        
        // The distance between the child and parent nodes serves as the length of the limb node geometry.
        let boxGeometry = SCNBox(width: CGFloat(Float(originalCubeWidth)),
                                 height: CGFloat(Float(length)),
                                 length: CGFloat(originalCubeWidth),
                                 chamferRadius: 0.05)
        node.geometry = boxGeometry
        node.geometry?.firstMaterial?.diffuse.contents = UIColor(ciColor: .red)
        
        // The node is positioned between the child and parent nodes.
        node.simdPosition = (toPoint + fromPoint) / 2
        node.simdEulerAngles = calculateLocalAngleToParent(joint: joint, fromPoint: fromPoint)
    }
    
    public func calculateLocalAngleToParent(joint: HumanKeypointName, fromPoint: simd_float3) -> simd_float3 {
        var angleVector: simd_float3 = simd_float3()
        do {
            
            let translationC  = fromPoint
            // The rotation for x, y, z.
            // Rotate 90 degrees from the default orientation of the node. Add yaw and pitch, and connect the child to the parent.
            let pitch = (Float.pi / 2)
            let yaw = acos(translationC.z / simd_length(translationC))
            let roll = atan2((translationC.y), (translationC.x))
            angleVector = simd_float3(pitch, yaw, roll)
            
        } catch {
            print("Unable to return point: \(error).")
        }
        return angleVector
    }
    
    func parentJointNameForJointName(jointName: HumanKeypointName) -> HumanKeypointName {
        switch jointName {
        case .Nose:
            return .Head
        case .Neck:
            return .Nose
        case .L_shoulder:
            return .Neck
        case .R_shoulder:
            return .Neck
        case .Torso:
            return .Neck
        case .L_Elbow:
            return .L_shoulder
        case .L_wrist:
            return .L_Elbow
        case .R_Elbow:
            return .R_shoulder
        case .R_wrist:
            return .R_Elbow
        case .Pelvis:
            return .Torso
        case .L_Hip:
            return .Pelvis
        case .R_Hip:
            return .Pelvis
        case .L_Knee:
            return .L_Hip
        case .R_Knee:
            return .R_Hip
        case .L_Ankle:
            return .L_Knee
        case .R_Ankle:
            return .R_Knee
        default:
            return .Head
        }
    }
}
