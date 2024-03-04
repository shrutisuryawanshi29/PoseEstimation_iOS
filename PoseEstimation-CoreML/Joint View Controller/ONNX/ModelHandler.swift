//
//  ModelHandler.swift
//  PoseEstimation-CoreML
//
//  Created by Shruti Suryawanshi on 9/27/23.
//  Copyright © 2023 tucan9389. All rights reserved.
//

import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import CoreML
import Vision
import TFLiteSwift_Vision

// Result struct
struct ResultOnnx {
    let processTimeMs: Double
    let inferences: [Inference]
}

// Inference struct for ssd model
struct Inference {
    let score: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

struct Prediction {
    let labelIndex: Int
    let confidence: Float
    let boundingBox: CGRect
    let pointArray: [NSValue]
    let visibleArray: [NSNumber]
}

// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

enum OrtModelError: Error {
    case error(_ message: String)
}

class ModelHandler: NSObject {
    // MARK: - Inference Properties

    let threadCount: Int32
    let threshold: Float = 0.5
    let threadCountLimit = 10
    
    // MARK: - Model Parameters

    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 192
    let inputHeight = 256
    
    let inputs = 34
    var delegate: DelegateForPAM?
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0 / 255.0, green: 200.0 / 255.0, blue: 250.0 / 255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    private var labels: [String] = []
    
    /// ORT inference session and environment object for performing inference on the given ssd model
    private var session: ORTSession
    private var env: ORTEnv
    
    // MARK: - Initialization of ModelHandler
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int32 = 1) {
        let modelFilename = modelFileInfo.name
        
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
        
        self.threadCount = threadCount
        do {
            // Start the ORT inference environment and specify the options for session
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
            try options.setIntraOpNumThreads(threadCount)
            // Create the ORTSession
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession.")
            return nil
        }
       
        super.init()
        
        //labels = loadLabels(fileInfo: labelsFileInfo)
    }

    // This method preprocesses the image, runs the ort inferencesession and returns the inference result
    func runModel(inputData: [CGFloat]) throws -> [Float32]{
        
        let inputName = "onnx::Reshape_0"
        
        let inputShape: [NSNumber] = [batchSize as NSNumber,
                                      
                                      inputs as NSNumber]
        
        do{
            
            // Prepare input data
            let interval: TimeInterval
            let inputNamee = try! session.inputNames()
            let inputShape: [Int] = [1, 34] // The shape of the input tensor
            let inputShape1: [NSNumber] = [1, 34] // The shape of the input tensor
            
            // Fill inputData with random values
//            for i in 0..<inputData.count {
//                inputData[i] = Float.random(in: 0.0..<1.0)
//            }
            
            // Convert input data to NSMutableData
            let inputTensorData = NSMutableData(bytes: inputData, length: inputData.count * MemoryLayout<Float>.size)
            
            // Create an ORTValue from the input data, element type, and shape
            let inputElementType: ORTTensorElementDataType = .float
            let inputValue = try! ORTValue(tensorData: inputTensorData, elementType: inputElementType, shape: inputShape1)
            var ortInputs: [String: ORTValue] = [inputName: inputValue]
            
            print("created input")
            
            // Run ORT InferenceSession
            let outputName = try! session.outputNames()
            let outputNames: Set<String> = ["2600","onnx::MatMul_2593"]
            
            //let startDate = Date()
            let startDate = Date()
            print(session.description)
            let outputs = try session.run(withInputs: ortInputs, outputNames: outputNames, runOptions: nil)
            // Process the output value as needed
            let outputTensorData = outputs
            let outputShape = outputs.count
            print(outputShape)
            
            print("ran the model")
            
            guard let rawOutputValue = outputs["2600"] else {
                throw OrtModelError1.error("failed to get model output_0")
            }
            let rawOutputData = try rawOutputValue.tensorData() as Data
            guard let outputArr: [Float32] = Array(unsafeData: rawOutputData) else {
                return []
            }
            print("The value of output is is \(outputArr)")
            print("the shape of output is \(outputArr.count)")
            
//            let reshapedArray = outputArr
//                .enumerated()
//                .map { (index, element) -> (Int, Int, Float) in
//                    let rowIndex = index / 3
//                    let columnIndex = index % 3
//                    return (rowIndex, columnIndex, element)
//                }
//                .reduce(into: [[Int]](repeating: [Int](repeating: 0, count: 3), count: 17)) { (result, element) in
//                    result[element.0][element.1] = Int(element.2)
//                }
//            
//            // Print the reshaped array
//            for row in reshapedArray {
//                print(row)
//            }
            
            interval = Date().timeIntervalSince(startDate) * 1000
            print("time taken = \(interval) ms")
            // let d = ORTTensorElementDataType <Float>(outputArr)
            delegate?.sendTheInference(inferenceTime: interval)
            guard let rawOutputValue_1 = outputs["onnx::MatMul_2593"] else {
                throw OrtModelError1.error("failed to get model output_1")
            }
            let rawOutputData_1 = try rawOutputValue_1.tensorData() as Data
            guard let outputArr_1: [Float32] = Array(unsafeData: rawOutputData_1) else {
                return []
            }
            
            return outputArr
        }
        catch let error {
            print("error in running data \(error.localizedDescription)")
        }
        return []
        
    }
    
    // MARK: - Helper Methods
    
    func convert1DArrayTo2DArray(array: [Float32], rows: Int, columns: Int) -> [[Float32]] {
        // Calculate the number of rows and columns in the 2D array.
        let numRows = array.count / columns
        let numColumns = columns

        // Create a new 2D array with the calculated dimensions.
        var twoDArray = Array(repeating: Array(repeating: Float32(), count: numColumns), count: numRows)

        // Iterate over the 1D array and add each element to the 2D array, row by row.
        var index = 0
        for row in 0..<numRows {
            for column in 0..<numColumns {
                twoDArray[row][column] = array[index]
                index += 1
            }
        }

        return twoDArray
    }
    
    func formatRTMPoseOutput(simccX: ORTValue, outputArr: [Float32], simccY: ORTValue, outputArr_1: [Float32]) {
        let simccxarr = try! simccX.tensorTypeAndShapeInfo().shape
        // Get the dimensions of the SimCC representations.
        let n = simccxarr[0]
        let k = simccxarr[1]
        let wx = simccxarr[2]
        
        let simx2d: [[Float32]] = convert1DArrayTo2DArray(array: outputArr, rows: Int(n)*Int(k), columns: Int(wx))
        
        let simccyarr = try! simccY.tensorTypeAndShapeInfo().shape
        // Get the dimensions of the SimCC representations.
        let ny = simccyarr[0]
        let ky = simccyarr[1]
        let wxy = simccyarr[2]
        
        let simy2d: [[Float32]] = convert1DArrayTo2DArray(array: outputArr_1, rows: Int(ny)*Int(ky), columns: Int(wxy))
        
        var xloc = [Int](repeating: 0, count:simx2d.count)
        var yloc = [Int](repeating: 0, count:simy2d.count)
        
        var maxxlocs = [Float32](repeating: 0, count:simx2d.count)
        var maxylocs = [Float32](repeating: 0, count:simy2d.count)
        
        for i in 0..<simx2d.count {
            maxxlocs[i] = simx2d[i].max()!
            xloc[i] = simx2d[i].firstIndex(of: maxxlocs[i]) ?? Array<Float32>.Index(0.0)
        }
        
        for i in 0..<simy2d.count {
            maxylocs[i] = simy2d[i].max()!
            yloc[i] = simy2d[i].firstIndex(of: maxylocs[i]) ?? Array<Float32>.Index(0.0)
        }
        
        
        var tuplexy: (Int, Int)
        
        for i in 0..<xloc.count {
            tuplexy.0 = xloc[i]
            tuplexy.1 = yloc[i]
        }
    }
    
    // This method postprocesses the results including processing bounding boxes, sort detected scores, etc.
    func formatResults(detectionBoxes: [Float32], detectionClasses: [Float32], detectionScores: [Float32],
                       numDetections: Int, width: CGFloat, height: CGFloat) -> [Inference]
    {
        var resultsArray: [Inference] = []
        
        if numDetections == 0 {
            return resultsArray
        }
        
        for i in 0 ..< numDetections {
            let score = detectionScores[i]
            
            // Filter results with score < threshold.
            guard score >= threshold else {
                continue
            }
            
            let detectionClassIndex = Int(detectionClasses[i])
            let detectionClass = labels[detectionClassIndex + 1]
            
            var rect = CGRect.zero
            
            // Translate the detected bounding box to CGRect.
            rect.origin.y = CGFloat(detectionBoxes[4 * i])
            rect.origin.x = CGFloat(detectionBoxes[4 * i + 1])
            rect.size.height = CGFloat(detectionBoxes[4 * i + 2]) - rect.origin.y
            rect.size.width = CGFloat(detectionBoxes[4 * i + 3]) - rect.origin.x
            
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            
            let colorToAssign = colorForClass(withIndex: detectionClassIndex + 1)
            let inference = Inference(score: score,
                                      className: detectionClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { first, second -> Bool in
            first.score > second.score
        }
        
        return resultsArray
    }
    
    // This method preprocesses the image by cropping pixel buffer to biggest square
    // and scaling the cropped image to model dimensions.
    private func preprocess(
        ofSize size: CGSize,
        _ buffer: CVPixelBuffer
    ) -> CVPixelBuffer? {
        let imageWidth = CVPixelBufferGetWidth(buffer)
        let imageHeight = CVPixelBufferGetHeight(buffer)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(buffer)
        
        assert(pixelBufferType == kCVPixelFormatType_32BGRA ||
            pixelBufferType == kCVPixelFormatType_32ARGB)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        // Find the biggest square in the pixel buffer and advance rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        // Get vImage_buffer
        var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress,
                                              height: UInt(imageHeight),
                                              width: UInt(imageWidth),
                                              rowBytes: inputImageRowBytes)
        
        let scaledRowBytes = Int(size.width) * imageChannels
        guard let scaledImageBytes = malloc(Int(size.height) * scaledRowBytes) else {
            return nil
        }
                
        var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes,
                                               height: UInt(size.height),
                                               width: UInt(size.width),
                                               rowBytes: scaledRowBytes)
        
        // Perform the scale operation on input image buffer and store it in scaled vImage buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))
        
        guard scaleError == kvImageNoError else {
            free(scaledImageBytes)
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = { _, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var scaledPixelBuffer: CVPixelBuffer?
        
        // Convert the scaled vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes,
            scaledRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer
        )
        
        guard conversionStatus == kCVReturnSuccess else {
            free(scaledImageBytes)
            return nil
        }
        
        return scaledPixelBuffer
    }
    
    private func loadLabels(fileInfo: FileInfo) -> [String] {
        var labelData: [String] = []
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            print("Labels file not found in bundle. Please add a labels file with name " +
                "\(filename).\(fileExtension)")
            return labelData
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labelData = contents.components(separatedBy: .newlines)
        } catch {
            print("Labels file named \(filename).\(fileExtension) cannot be read.")
        }

        return labelData
    }
    
    private func colorForClass(withIndex index: Int) -> UIColor {
        // Assign variations to the base colors for each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((10 / 2 - index / colors.count) * 10)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
    
    // Return the RGB data representation of the given image buffer.
    func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool = true
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
        
        switch pixelBufferFormat {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        
        return byteData
    }
    
    func parsePoseYOLOv8(results: [VNCoreMLFeatureValueObservation], modelType: ModelType) -> [Prediction] {
        guard let observation = results.first else { return [] }
        guard let outputArray = observation.featureValue.multiArrayValue else { return [] }

        var result: Result<PoseEstimationOutput, PoseEstimationError>
        
        let gridHeight = outputArray.shape[1].intValue
        let gridWidth = outputArray.shape[2].intValue
        let classesNum = gridHeight - 4 - 17*3//self.keypointsNum * self.keypointsDim
        let threshold: Float = 0.25//self.confidenceThreshold
        let inputWidth = self.inputWidth

        var predictions = [Prediction]()

        for j in 0..<gridWidth {
            var classIndex = -1
            var maxScore: Float = 0.0

            for i in 4..<4 + classesNum {
                let score = outputArray[(i * gridWidth) + j].floatValue
                if score > maxScore {
                    classIndex = i - 4
                    maxScore = score
                 }
            }

            if maxScore > threshold {
                var x: Float = outputArray[(0 * gridWidth) + j].floatValue
                var y = outputArray[(1 * gridWidth) + j].floatValue
                var w = outputArray[(2 * gridWidth) + j].floatValue
                var h = outputArray[(3 * gridWidth) + j].floatValue

                x -= w/2
                y -= h/2
                let cgrectX = CGFloat(x/Float(inputWidth))
                let cgrectY = CGFloat(1.0 - (y + h) / Float(inputWidth))
                let cgrectW = CGFloat(w / Float(inputWidth))
                let cgrectH = CGFloat(h / Float(inputWidth))
                
                let rect = CGRect(x: cgrectX, y: cgrectY, width: cgrectW, height: cgrectH)
                
                var pointArray = [NSValue]()
                var visibleArray = [NSNumber]()
                
                var twodArrayForYolo = [[CGFloat]]()

                for i in stride(from: 4 + classesNum, to: gridHeight, by: 3){
                    let x = outputArray[((i + 0) * gridWidth) + j].floatValue
                    let y = outputArray[((i + 1) * gridWidth) + j].floatValue
                    let v = outputArray[((i + 2) * gridWidth) + j].floatValue

                    let point = CGPoint(x: Double(x / Float(inputWidth)), y: Double(1.0 - y / Float(inputWidth)))
                    
                    //let visible = v > 0 ? kVisible : kNotVisibleNotLabeled

                    twodArrayForYolo.append([point.x, point.y])
                    pointArray.append(NSValue(cgPoint: point))
                    //visibleArray.append(NSNumber(integerLiteral: visible))
                }
                
                if modelType == .PAM {
                    print("1D array - \(convert2DArrayto1D(twoDArray: twodArrayForYolo))")
                    
                    // Record the start time
                    let startTime = DispatchTime.now()
                    
                    var outputs = try! runModel(inputData: convert2DArrayto1D(twoDArray: twodArrayForYolo))
                    
                    // Record the end time
                    let endTime = DispatchTime.now()

                    // Calculate the execution time in nanoseconds
                    let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds

                    // Convert the execution time to milliseconds
                    let executionTimeInMilliseconds = Double(nanoTime) / 1_000_000

                    print("PAM Execution Time: \(executionTimeInMilliseconds) milliseconds")
                    
                    //return outputs
                }
                
                let prediction = Prediction(labelIndex: classIndex,
                                            confidence: maxScore,
                                            boundingBox: rect,
                                            pointArray: pointArray,
                                            visibleArray: visibleArray)
                

                predictions.append(prediction)
            }
        }

        predictions = getPredictionsNMS(predictions: predictions)
        
        return predictions
    }
    
    func parsePoseYOLOv8ForPAM(results: [VNCoreMLFeatureValueObservation], modelType: ModelType) -> [Float32] {
        guard let observation = results.first else { return [] }
        guard let outputArray = observation.featureValue.multiArrayValue else { return [] }

        var result: Result<PoseEstimationOutput, PoseEstimationError>
        
        let gridHeight = outputArray.shape[1].intValue
        let gridWidth = outputArray.shape[2].intValue
        let classesNum = gridHeight - 4 - 17*3//self.keypointsNum * self.keypointsDim
        let threshold: Float = 0.25//self.confidenceThreshold
        let inputWidth = self.inputWidth

        var predictions = [Prediction]()

        for j in 0..<gridWidth {
            var classIndex = -1
            var maxScore: Float = 0.0

            for i in 4..<4 + classesNum {
                let score = outputArray[(i * gridWidth) + j].floatValue
                if score > maxScore {
                    classIndex = i - 4
                    maxScore = score
                 }
            }

            if maxScore > threshold {
                var x: Float = outputArray[(0 * gridWidth) + j].floatValue
                var y = outputArray[(1 * gridWidth) + j].floatValue
                var w = outputArray[(2 * gridWidth) + j].floatValue
                var h = outputArray[(3 * gridWidth) + j].floatValue

                x -= w/2
                y -= h/2
                let cgrectX = CGFloat(x/Float(inputWidth))
                let cgrectY = CGFloat(1.0 - (y + h) / Float(inputWidth))
                let cgrectW = CGFloat(w / Float(inputWidth))
                let cgrectH = CGFloat(h / Float(inputWidth))
                
                let rect = CGRect(x: cgrectX, y: cgrectY, width: cgrectW, height: cgrectH)
                
                var pointArray = [NSValue]()
                var visibleArray = [NSNumber]()
                
                var twodArrayForYolo = [[CGFloat]]()

                for i in stride(from: 4 + classesNum, to: gridHeight, by: 3){
                    let x = outputArray[((i + 0) * gridWidth) + j].floatValue
                    let y = outputArray[((i + 1) * gridWidth) + j].floatValue
                    let v = outputArray[((i + 2) * gridWidth) + j].floatValue

                    let point = CGPoint(x: Double(x / Float(inputWidth)), y: Double(1.0 - y / Float(inputWidth)))
                    
                    //let visible = v > 0 ? kVisible : kNotVisibleNotLabeled

                    twodArrayForYolo.append([point.x, point.y])
                    pointArray.append(NSValue(cgPoint: point))
                    //visibleArray.append(NSNumber(integerLiteral: visible))
                }
                
                if modelType == .PAM {
                    print("1D array - \(convert2DArrayto1D(twoDArray: twodArrayForYolo))")
                    
                    // Record the start time
                    let startTime = DispatchTime.now()
                    
                    var outputs = try! runModel(inputData: convert2DArrayto1D(twoDArray: twodArrayForYolo))
                    
                    // Record the end time
                    let endTime = DispatchTime.now()

                    // Calculate the execution time in nanoseconds
                    let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds

                    // Convert the execution time to milliseconds
                    let executionTimeInMilliseconds = Double(nanoTime) / 1_000_000

                    print("PAM Execution Time: \(executionTimeInMilliseconds) milliseconds")
                    
                    return outputs
                }
                
                let prediction = Prediction(labelIndex: classIndex,
                                            confidence: maxScore,
                                            boundingBox: rect,
                                            pointArray: pointArray,
                                            visibleArray: visibleArray)
                

                predictions.append(prediction)
            }
        }

        predictions = getPredictionsNMS(predictions: predictions)
        
        return []
    }

    func getPredictionsNMS(predictions: [Prediction]) -> [Prediction] {
        guard predictions.count >= 2 else { return predictions }

        let sortedPredictions = predictions.sorted { $0.confidence > $1.confidence }
        var keep = [Bool](repeating: true, count: sortedPredictions.count)
        var predictionsNMS = [Prediction]()

        for i in 0..<sortedPredictions.count {
            if keep[i] {
                predictionsNMS.append(sortedPredictions[i])

                let prediction = sortedPredictions[i]
                let bbox1 = prediction.boundingBox

                for j in i + 1..<sortedPredictions.count {
                    if keep[j] {
                        let predictionJ = sortedPredictions[j]
                        let bbox2 = predictionJ.boundingBox
                        let threshold: Float = 0.5//self.nmsThreshold
                        let iou = IoU(rect1: bbox1, rect2: bbox2)

                        if iou > threshold {
                            keep[j] = false
                        }
                    }
                }
            }
        }

        return predictionsNMS
    }

    func IoU(rect1: CGRect, rect2: CGRect) -> Float {
//        let intersectionRect = rect1.intersection(rect2)
//        let unionRect = rect1.union(rect2)
//        
//        if intersectionRect.isEmpty {
//            return 0.0
//        }
//        
//        let iou = intersectionRect.width * intersectionRect.height / (unionRect.width * unionRect.height)
//        
//        return Float(iou)
        
        let areaA = rect1.size.width * rect1.size.height
        if areaA <= 0 { return 0 }
        
        let areaB = rect2.size.width * rect2.size.height
        if areaB <= 0 { return 0 }
        
        let intersectionMinX = max(rect1.minX, rect2.minX)
        let intersectionMinY = max(rect1.minY, rect2.minY)
        let intersectionMaxX = min(rect1.maxX, rect2.maxX)
        let intersectionMaxY = min(rect1.maxY, rect2.maxY)
        
        let intersectionWidth = max(intersectionMaxX - intersectionMinX, 0)
        let intersectionHeight = max(intersectionMaxY - intersectionMinY, 0)
        
        let intersectionArea = intersectionWidth * intersectionHeight
        return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }
    
    func convert2DArrayto1D(twoDArray: [[CGFloat]]) -> [CGFloat]{
        // Initialize an empty 1D array
        var oneDArray: [CGFloat] = []

        // Iterate through the 2D array and append elements to the 1D array
        for row in twoDArray {
            for element in row {
                oneDArray.append(element)
            }
        }
        
        return oneDArray
    }
}

// MARK: - Extensions

extension ModelHandler {
    
    
    
    func postprocess(with outputs: [TFLiteFlatArray]) -> PoseEstimationOutput {
        return PoseEstimationOutput(outputs: outputs)
    }
}

extension Data {
    // Create a new buffer by copying the buffer pointer of the given array.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    // Create a new array from the bytes of the given unsafe data.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
        #if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
        #else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
        #endif // swift(>=5.0)
    }
}

extension UIColor {
    // This method returns colors modified by percentage value of color represented by the current object.
    func getModified(byPercentage percent: CGFloat) -> UIColor? {
        var red: CGFloat = 0.0
        var green: CGFloat = 0.0
        var blue: CGFloat = 0.0
        var alpha: CGFloat = 0.0
        
        guard getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return nil
        }
        
        // Return the color comprised by percentage r g b values of the original color.
        let colorToReturn = UIColor(displayP3Red: min(red + percent / 100.0, 1.0),
                                    green: min(green + percent / 100.0, 1.0),
                                    blue: min(blue + percent / 100.0, 1.0),
                                    alpha: 1.0)
        
        return colorToReturn
    }
}

extension RangeReplaceableCollection {
    public mutating func resize(_ size: IndexDistance, fillWith value: Iterator.Element) {
        let c = count
        if c < size {
            append(contentsOf: repeatElement(value, count: c.distance(to: size)))
        } else if c > size {
            let newEnd = index(startIndex, offsetBy: size)
            removeSubrange(newEnd ..< endIndex)
        }
    }
}

