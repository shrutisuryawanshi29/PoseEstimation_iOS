//
//  HumanKeypoints.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2021/07/11.
//  Copyright © 2021 Doyoung Gwak. All rights reserved.
//

import CoreGraphics
import TFLiteSwift_Vision
import simd

class Keypoint {
    
    let x: Float
    let y: Float
    let z: Float?
    let score: Float
    var is3D: Bool { return z != nil }
    
    init(x: Float, y: Float, z: Float?, s: Float) {
        self.x = x
        self.y = y
        self.z = z
        self.score = s
    }
    
    var location2D: CGPoint {
        return CGPoint(x: CGFloat(x), y: CGFloat(y))
    }
    
    var yFlip: Keypoint {
        return Keypoint(x: x, y: 1-y, z: z, s: score)
    }
}

enum HumanKeypointName: Int, CaseIterable {
    case Pelvis      = 0
      case R_Hip      = 1
      case R_Knee = 2
      case R_Ankle   = 3
      case L_Hip   = 4
      case L_Knee  = 5
      case L_Ankle   = 6
      case Torso   = 7
      case Neck    = 8
      case Nose   = 9
      case Head   = 10
      case L_shoulder    = 11
      case L_Elbow   = 12
      case L_wrist   = 13
      case R_shoulder    = 14
      case R_Elbow     = 15
      case R_wrist   = 16
}

class HumanKeypoints {
    
    let keypoints: [Keypoint?]
    var is3D: Bool { return keypoints.first??.is3D == true }
    
    typealias Line = (from: Keypoint?, to: Keypoint?)
    static var lineInfos: [(from: HumanKeypointName, to: HumanKeypointName)] = [
        (.Pelvis, .R_Hip),
        (.R_Hip, .R_Knee),
        (.R_Knee, .R_Ankle),
        (.Pelvis, .L_Hip),
        (.L_Hip, .L_Knee),
        (.L_Knee, .L_Ankle),
        (.Pelvis, .Torso),
        (.Torso, .Neck),
        (.Neck, .Nose),
        (.Nose, .Head),
        (.Neck, .L_shoulder),
        (.L_shoulder, .L_Elbow),
        (.L_Elbow, .L_wrist),
        (.Neck, .R_shoulder),
        (.R_shoulder, .R_Elbow),
        (.L_Elbow, .R_wrist),
      ]
    var lines: [Line] {
        return HumanKeypoints.lineInfos.map { (keypoints[$0.from.rawValue], keypoints[$0.to.rawValue]) }
    }
    
    init(keypoints: [Keypoint?]) {
        self.keypoints = keypoints
    }
    
    init(human3d: PoseEstimationOutput.Human3D, adjustMode: Bool = false) {
        let allParts = HumanKeypoints.Output.BodyPart.allCases
        let partToIndex: [HumanKeypoints.Output.BodyPart: Int] = Dictionary(uniqueKeysWithValues: allParts.enumerated().map { ($0.element, $0.offset) })
        let jointParts: [HumanKeypoints.Output.BodyPart] = [
//            .HEAD,
//            .THORAX,
//            
//            .RIGHT_SHOULDER,
//            .RIGHT_ELBOW,
//            .RIGHT_WRIST,
//            .LEFT_SHOULDER,
//            .LEFT_ELBOW,
//            .LEFT_WRIST,
//            
//            .RIGHT_HIP,
//            .RIGHT_KNEE,
//            .RIGHT_ANKLE,
//            .LEFT_HIP,
//            .LEFT_KNEE,
//            .LEFT_ANKLE,
            
            .Head,
            .R_shoulder,
            .R_Elbow,
            .R_wrist,
            .L_shoulder,
            .L_Elbow,
            .L_wrist,
            .R_Hip,
            .R_Knee,
            .R_Ankle,
            .L_Hip,
            .L_Knee,
            .L_Ankle,
        ]
        
        let kps = adjustMode ? human3d.keypoints : human3d.adjustKeypoints()
        self.keypoints = jointParts.map {
            guard let kpIndex = partToIndex[$0], let point = kps[kpIndex] else { return nil }
            return Keypoint(x: Float(point.position.x), y: Float(point.position.y), z: Float(point.position.z), s: point.score)
        }
    }
    
  /*
    init(mlkitPose: Pose, imageSize: CGSize) {
        let jointParts: [PoseLandmarkType] = [
            .nose,
            .nose,  // need to average value of rightShoulder and leftShoulde
            
            .rightShoulder,
            .rightElbow,
            .rightWrist,
            .leftShoulder,
            .leftElbow,
            .leftWrist,
            
            .rightHip,
            .rightKnee,
            .rightAnkle,
            .leftHip,
            .leftKnee,
            .leftAnkle,
        ]
        
        let maxZ: CGFloat = (imageSize.height + imageSize.width) * 0.8 // CGFloat.leastNormalMagnitude
        let minZ: CGFloat = -maxZ // CGFloat.greatestFiniteMagnitude
//        jointParts.forEach {
//            let position = mlkitPose.landmark(ofType: $0).position
//            maxZ = max(position.z, maxZ)
//            minZ = min(position.z, minZ)
//        }
        let zDistance = abs(maxZ - minZ)
//        minZ -= zDistance*2
//        maxZ += zDistance*2
//        zDistance = abs(maxZ - minZ)
//
//        print(maxZ, minZ, imageSize)
        
        self.keypoints = jointParts.enumerated().map {
            if $0 == 1 {
                let landmark1 = mlkitPose.landmark(ofType: .rightShoulder)
                let landmark2 = mlkitPose.landmark(ofType: .leftShoulder)
                let absoluteZ = (landmark1.position.z + landmark2.position.z) / 2
                let x = (landmark1.position.x + landmark2.position.x) / 2
                let y = (landmark1.position.y + landmark2.position.y) / 2
                let z = Float((absoluteZ - minZ) / zDistance)
                let s = (landmark1.inFrameLikelihood + landmark2.inFrameLikelihood) / 2
                return Keypoint(x: Float(x / imageSize.width), y: Float(1 - (y / imageSize.height)), z: z, s: s)
            } else {
                let position = mlkitPose.landmark(ofType: $1).position
                let z = Float((position.z - minZ) / zDistance)
                return Keypoint(x: Float(position.x / imageSize.width), y: Float(1 - (position.y / imageSize.height)), z: z, s: mlkitPose.landmark(ofType: $1).inFrameLikelihood)
            }
        }
    }
   */
    
    init(vision2DPoseKeypoints: HumanKeypoints, mlkit3DPoseKeypoints: HumanKeypoints) {
        let zipOfPoints = zip(vision2DPoseKeypoints.keypoints, mlkit3DPoseKeypoints.keypoints)
        self.keypoints = zipOfPoints.enumerated().map {
            if let kp2D = $1.0, let kp3D = $1.1 {
                let z: Float?
                if $0 == HumanKeypointName.Neck.rawValue {
                    z = ((mlkit3DPoseKeypoints.keypoints[HumanKeypointName.R_shoulder.rawValue]?.z ?? 0) + (mlkit3DPoseKeypoints.keypoints[HumanKeypointName.L_shoulder.rawValue]?.z ?? 0)) / 2
                } else {
                    z = kp3D.z
                }
                if kp2D.score < 0.4 {
                    return Keypoint(x: kp3D.x, y: kp3D.y, z: z, s: kp3D.score)
                } else {
                    return Keypoint(x: kp2D.x, y: kp2D.y, z: z, s: kp3D.score)
                }
            }
            
            return nil
        }
    }
    
//    var predictedPoints: [PredictedPoint?] {
//        return keypoints.map {
//            guard let keypoint = $0 else { return nil }
//            return PredictedPoint(maxPoint: CGPoint(x: CGFloat(keypoint.x), y: CGFloat(keypoint.y)), maxConfidence: Double(keypoint.score))
//        }
//    }
    
    subscript(keypointName: HumanKeypointName) -> Keypoint? {
        let kpIndex = keypointName.rawValue
        guard (0..<keypoints.count).contains(kpIndex) else { return nil }
        return keypoints[kpIndex]
    }
    
    init(kpsArray: [HumanKeypoints]) {
        guard let firstKeypoints = kpsArray.first else { self.keypoints = []; return }
        self.keypoints = (0..<firstKeypoints.keypoints.count).map { idx in
            guard let x: Float = kpsArray[(kpsArray.count-1)/2].keypoints[idx]?.x,
                  let y: Float = kpsArray[(kpsArray.count-1)/2].keypoints[idx]?.y else { return nil }
            let z: Float = kpsArray.compactMap { $0.keypoints[idx]?.z }.reduce(0) { $0 + $1 } / Float((kpsArray.compactMap { $0.keypoints[idx]?.z }).count)
            let s: Float = kpsArray.compactMap { $0.keypoints[idx]?.score }.reduce(0) { $0 + $1 } / Float((kpsArray.compactMap { $0.keypoints[idx]?.score }).count)
            return Keypoint(x: x, y: y, z: z, s: s)
        }
    }
}

extension HumanKeypoints {
    struct Output {
        struct Heatmap {
            static let width = 32
            static let height = 32
            static let depth = 32
            static let count = BodyPart.allCases.count // 18
        }
        
        enum BodyPart: String, CaseIterable {
//            case HEAD_TOP = "Head_top"              // 0
//            case THORAX = "Thorax"                  // 1
//            case RIGHT_SHOULDER = "R_Shoulder"      // 2
//            case RIGHT_ELBOW = "R_Elbow"            // 3
//            case RIGHT_WRIST = "R_Wrist"            // 4
//            case LEFT_SHOULDER = "L_Shoulder"       // 5
//            case LEFT_ELBOW = "L_Elbow"             // 6
//            case LEFT_WRIST = "L_Wrist"             // 7
//            case RIGHT_HIP = "R_Hip"                // 8
//            case RIGHT_KNEE = "R_Knee"              // 9
//            case RIGHT_ANKLE = "R_Ankle"            // 10
//            case LEFT_HIP = "L_Hip"                 // 11
//            case LEFT_KNEE = "L_Knee"               // 12
//            case LEFT_ANKLE = "L_Ankle"             // 13
//            case PELVIS = "Pelvis"                  // 14
//            case SPINE = "Spine"                    // 15
//            case HEAD = "Head"                      // 16
//            case RIGHT_HAND = "R_Hand"              // 17
//            case LEFT_HAND = "L_Hand"               // 18
//            case RIGHT_TOE = "R_Toe"                // 19
//            case LEFT_TOE = "L_Toe"                // 20
//            
            case Pelvis      = "Pelvis"
              case R_Hip      = "R_Hip"
              case R_Knee = "R_Knee"
              case R_Ankle   = "R_Ankle"
              case L_Hip   = "L_Hip"
              case L_Knee  = "L_Knee"
              case L_Ankle   = "L_Ankle"
              case Torso   = "Torso"
              case Neck    = "Neck"
              case Nose   = "Nose"
              case Head   = "Head"
              case L_shoulder    = "L_shoulder"
              case L_Elbow   = "L_Elbow"
              case L_wrist   = "L_wrist"
              case R_shoulder    = "R_shoulder"
              case R_Elbow     = "R_Elbow"
              case R_wrist   = "R_wrist"
            
            static let baselineKeypointIndexes = (2, 5)  // R_Shoulder, L_Shoulder

            static let lines = [
                (.Pelvis, .R_Hip),
                (.R_Hip, .R_Knee),
                (.R_Knee, .R_Ankle),
                (.Pelvis, .L_Hip),
                (.L_Hip, .L_Knee),
                (.L_Knee, .L_Ankle),
                (.Pelvis, .Torso),
                (.Torso, .Neck),
                (.Neck, .Nose),
                (.Nose, .Head),
                (.Neck, .L_shoulder),
                (.L_shoulder, .L_Elbow),
                (.L_Elbow, .L_wrist),
                (.Neck, .R_shoulder),
                (.R_shoulder, .R_Elbow),
                (.L_Elbow, .R_wrist),
                
                (from: BodyPart.Pelvis, to: BodyPart.R_Hip),
                (from: BodyPart.R_Hip, to: BodyPart.R_Knee),
                (from: BodyPart.R_Knee, to: BodyPart.R_Ankle),
                (from: BodyPart.Pelvis, to: BodyPart.L_Hip),
                (from: BodyPart.L_Hip, to: BodyPart.L_Knee),
                (from: BodyPart.L_Knee, to: BodyPart.L_Ankle),
                (from: BodyPart.Pelvis, to: BodyPart.Torso),
                (from: BodyPart.Torso, to: BodyPart.Neck),
                (from: BodyPart.Neck, to: BodyPart.Nose),
                (from: BodyPart.Nose, to: BodyPart.Head),
                (from: BodyPart.Neck, to: BodyPart.L_shoulder),
                (from: BodyPart.L_shoulder, to: BodyPart.L_Elbow),
                (from: BodyPart.L_Elbow, to: BodyPart.L_wrist),
                (from: BodyPart.Neck, to: BodyPart.R_shoulder),
//                (from: BodyPart.RIGHT_SHOULDER, to: BodyPart.RIGHT_ELBOW),
//                (from: BodyPart.RIGHT_ELBOW, to: BodyPart.RIGHT_WRIST),
//                (from: BodyPart.RIGHT_WRIST, to: BodyPart.RIGHT_HAND),
//                (from: BodyPart.THORAX, to: BodyPart.LEFT_SHOULDER),
//                (from: BodyPart.LEFT_SHOULDER, to: BodyPart.LEFT_ELBOW),
//                (from: BodyPart.LEFT_ELBOW, to: BodyPart.LEFT_WRIST),
//                (from: BodyPart.LEFT_WRIST, to: BodyPart.LEFT_HAND),
            ]
        }
    }
}

extension CGRect {
    func iou(with rect: CGRect) -> CGFloat {
        let r1 = self
        let r2 = rect
        let x1 = max(r1.origin.x, r2.origin.x)
        let x2 = min(r1.origin.x + r1.width, r2.origin.x + r2.width)
        guard x1 < x2 else { return 0 }
        let y1 = max(r1.origin.y, r2.origin.y)
        let y2 = min(r1.origin.y + r1.height, r2.origin.y + r2.height)
        guard y1 < y2 else { return 0 }
        let intersactionArea = (x2 - x1) * (y2 - y1)
        let unionArea = (r1.width * r1.height) + (r2.width * r2.height) - intersactionArea
        return intersactionArea / unionArea
    }
    
    var center: CGPoint {
        return CGPoint(x: origin.x + width/2, y: origin.y + height/2)
    }
    
    var squareRect: CGRect {
        let longLength = width > height ? width : height
        return CGRect(x: center.x - longLength / 2, y: center.y - longLength / 2, width: longLength, height: longLength)
    }
    
    static func * (_ lsh: CGRect, rsh: CGSize) -> CGRect {
        return CGRect(x: lsh.origin.x * rsh.width, y: lsh.origin.y * rsh.height, width: lsh.width * rsh.width, height: lsh.height * rsh.height)
    }
    
    var yFlip: CGRect {
        return CGRect(x: origin.x, y: 1 - origin.y - height, width: width, height: height)
    }
    
    func adjustAsInside(parentSize: CGSize) -> CGRect {
        let x: CGFloat
        if origin.x < 0 {
            x = 0
        } else {
            x = origin.x
        }
        let w: CGFloat
        if x + width > parentSize.width {
            w = parentSize.width - x
        } else {
            w = width
        }
        let y: CGFloat
        if origin.y < 0 {
            y = 0
        } else {
            y = origin.y
        }
        let h: CGFloat
        if y + height > parentSize.height {
            h = parentSize.height - y
        } else {
            h = height
        }
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

struct PoseEstimationOutput {
    
    struct Human3D {
        typealias Line3D = (from: Keypoint3D, to: Keypoint3D)
        var keypoints: [Keypoint3D?] = []
        var lines: [Line3D] = []
        var baselineKeypointIndexes: (Int, Int)? = nil
    }
    
    enum Human {
        case human3d(human: Human3D)
        
        var human3d: Human3D? {
            if case .human3d(let human) = self {
                return human
            } else { return nil }
        }
    }
    
    var outputs: [TFLiteFlatArray]
    var humans: [Human] = []
    var humans3d: [Human3D?] { return humans.map { $0.human3d } }
}

enum PoseEstimationError: Error {
    case failToCreateInputData
    case failToInference
    case failToPostprocess
}

struct Keypoint3D {
    
    struct Point3D {
        let x: CGFloat
        let y: CGFloat
        let z: CGFloat
        
        var simdVector: simd_float3 {
            return simd_float3(x: Float(x), y: Float(y), z: Float(z))
        }
    }
    
    let position: Point3D
    let score: Float
    
    init(x: CGFloat, y: CGFloat, z: CGFloat, s: Float = 1.0) {
        position = Point3D(x: x, y: y, z: z)
        score = s
    }
    
    static func - (lhs: Keypoint3D, rhs: Keypoint3D) -> Keypoint3D {
        return Keypoint3D(
            x: lhs.position.x - rhs.position.x,
            y: lhs.position.y - rhs.position.y,
            z: lhs.position.z - rhs.position.z
        )
    }
    static func + (lhs: Keypoint3D, rhs: Keypoint3D) -> Keypoint3D {
        return Keypoint3D(
            x: lhs.position.x + rhs.position.x,
            y: lhs.position.y + rhs.position.y,
            z: lhs.position.z + rhs.position.z
        )
    }
    static func * (lhs: Keypoint3D, rhs: Keypoint3D) -> Keypoint3D {
        return Keypoint3D(
            x: lhs.position.x * rhs.position.x,
            y: lhs.position.y * rhs.position.y,
            z: lhs.position.z * rhs.position.z
        )
    }
    static func / (lhs: Keypoint3D, rhs: Keypoint3D) -> Keypoint3D {
        return Keypoint3D(
            x: lhs.position.x / rhs.position.x,
            y: lhs.position.y / rhs.position.y,
            z: lhs.position.z / rhs.position.z
        )
    }
    var distance: CGFloat {
        return pow(position.x*position.x + position.y*position.y + position.z*position.z, 0.5)
    }
    
    func product(rhs: Keypoint3D) -> CGFloat {
        let v = self * rhs
        return v.position.x + v.position.y + v.position.z
    }
}
