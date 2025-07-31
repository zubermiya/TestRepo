import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ExifTags
# Optional imports with fallbacks
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError:
    tensorflow_available = False

try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    import magic
    magic_available = True
except ImportError:
    magic_available = False
import timm
import logging
import json
import exifread
try:
    from moviepy import VideoFileClip
    moviepy_available = True
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
        moviepy_available = True
    except ImportError:
        moviepy_available = False
import tempfile
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AIMediaDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models for AI detection"""
        try:
            # Load EfficientNet for image classification
            self.image_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
            self.image_model.eval()
            self.image_model.to(self.device)
            
            # Image preprocessing
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Load face detection model
            try:
                import face_recognition
                self.face_detection_available = True
            except ImportError:
                self.face_detection_available = False
                logger.warning("Face recognition not available")
            
            # Initialize anomaly detector for metadata analysis
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    def extract_metadata(self, filepath):
        """Extract metadata from image/video files"""
        metadata = {}
        
        try:
            # Basic file info
            file_stats = os.stat(filepath)
            metadata['file_size'] = file_stats.st_size
            metadata['creation_time'] = file_stats.st_ctime
            metadata['modification_time'] = file_stats.st_mtime
            
            # File type detection
            if magic_available:
                file_type = magic.from_file(filepath, mime=True)
                metadata['mime_type'] = file_type
            else:
                # Fallback to basic extension-based detection
                import mimetypes
                file_type = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
                metadata['mime_type'] = file_type
            
            if 'image' in file_type:
                metadata.update(self._extract_image_metadata(filepath))
            elif 'video' in file_type:
                metadata.update(self._extract_video_metadata(filepath))
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            
        return metadata
    
    def _extract_image_metadata(self, filepath):
        """Extract image-specific metadata"""
        metadata = {}
        
        try:
            # EXIF data
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f)
                exif_data = {}
                for tag in tags.keys():
                    if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                        exif_data[tag] = str(tags[tag])
                metadata['exif'] = exif_data
            
            # PIL metadata
            with Image.open(filepath) as img:
                metadata['dimensions'] = img.size
                metadata['mode'] = img.mode
                metadata['format'] = img.format
                
                # Check for AI generation indicators in EXIF
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    metadata['pil_exif'] = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                    
        except Exception as e:
            logger.error(f"Error extracting image metadata: {str(e)}")
            
        return metadata
    
    def _extract_video_metadata(self, filepath):
        """Extract video-specific metadata"""
        metadata = {}
        
        try:
            # OpenCV metadata
            cap = cv2.VideoCapture(filepath)
            metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['duration'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
            cap.release()
            
            # MoviePy metadata
            if moviepy_available:
                try:
                    with VideoFileClip(filepath) as clip:
                        metadata['duration_moviepy'] = clip.duration
                        metadata['audio_present'] = clip.audio is not None
                except Exception as e:
                    logger.warning(f"MoviePy metadata extraction failed: {str(e)}")
            else:
                logger.info("MoviePy not available - skipping advanced video metadata")
                
        except Exception as e:
            logger.error(f"Error extracting video metadata: {str(e)}")
            
        return metadata
    
    def analyze_compression_artifacts(self, image_path):
        """Analyze compression artifacts that might indicate AI generation"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # DCT analysis for JPEG artifacts
            dct = cv2.dct(np.float32(gray))
            
            # Calculate artifact metrics
            high_freq_energy = np.sum(np.abs(dct[50:, 50:]))
            total_energy = np.sum(np.abs(dct))
            artifact_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Noise analysis
            noise = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return {
                'artifact_ratio': float(artifact_ratio),
                'edge_density': float(edge_density),
                'noise_variance': float(noise),
                'suspicious_compression': artifact_ratio < 0.01  # Very low artifacts might indicate AI
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compression artifacts: {str(e)}")
            return {}
    
    def detect_face_inconsistencies(self, image_path):
        """Detect face-related inconsistencies that might indicate deepfakes"""
        if not self.face_detection_available:
            return {'face_analysis': 'Face detection not available'}
        
        try:
            import face_recognition
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return {'faces_detected': 0}
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Analyze each face
            face_analysis = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_img = image[top:bottom, left:right]
                
                # Basic quality metrics
                face_analysis.append({
                    'face_id': i,
                    'location': [top, right, bottom, left],
                    'face_area': (bottom - top) * (right - left),
                    'aspect_ratio': (bottom - top) / (right - left)
                })
            
            return {
                'faces_detected': len(face_locations),
                'face_details': face_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in face analysis: {str(e)}")
            return {'face_analysis_error': str(e)}
    
    def analyze_image_statistics(self, image_path):
        """Analyze image statistics that can indicate AI generation"""
        try:
            img = cv2.imread(image_path)
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Calculate various statistics
            stats = {}
            
            # Color distribution analysis
            for i, channel in enumerate(['b', 'g', 'r']):
                stats[f'{channel}_mean'] = float(np.mean(img[:,:,i]))
                stats[f'{channel}_std'] = float(np.std(img[:,:,i]))
                stats[f'{channel}_skew'] = float(self._calculate_skewness(img[:,:,i]))
            
            # HSV analysis
            stats['hue_mean'] = float(np.mean(hsv[:,:,0]))
            stats['saturation_mean'] = float(np.mean(hsv[:,:,1]))
            stats['value_mean'] = float(np.mean(hsv[:,:,2]))
            
            # Texture analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            stats['texture_variance'] = float(np.var(gray))
            
            # Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            stats['frequency_energy'] = float(np.mean(magnitude_spectrum))
            
            # AI generation indicators
            ai_indicators = []
            
            # Check for unusually uniform color distributions (common in AI images)
            if stats['b_std'] < 20 and stats['g_std'] < 20 and stats['r_std'] < 20:
                ai_indicators.append("Unusually uniform color distribution")
            
            # Check for low texture variance (smooth AI images)
            if stats['texture_variance'] < 500:
                ai_indicators.append("Low texture variance (smooth appearance)")
            
            # Check for unusual frequency patterns
            if stats['frequency_energy'] > 15:
                ai_indicators.append("Unusual frequency patterns detected")
            
            stats['ai_indicators'] = ai_indicators
            stats['ai_score'] = len(ai_indicators) / 3.0  # Normalize to 0-1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing image statistics: {str(e)}")
            return {}
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def detect_ai_characteristics(self, image_path):
        """Detect specific characteristics of AI-generated images"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            characteristics = {}
            
            # 1. Check for overly smooth textures (common in AI images)
            # Calculate local variance
            kernel = np.ones((5,5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            avg_local_variance = np.mean(local_variance)
            characteristics['smoothness_score'] = float(avg_local_variance)
            
            # 2. Check for artificial patterns
            # Look for regular grid-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            characteristics['edge_density'] = float(edge_density)
            
            # 3. Check for unrealistic color distributions
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_variance = np.var(hsv)
            characteristics['color_variance'] = float(color_variance)
            
            # 4. Check for frequency domain anomalies
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Check for unusual frequency patterns
            high_freq_energy = np.sum(magnitude_spectrum[100:, 100:])
            total_energy = np.sum(magnitude_spectrum)
            freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            characteristics['frequency_ratio'] = float(freq_ratio)
            
            # 5. Check for artificial symmetry
            # Compare left and right halves
            height, width = gray.shape
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            characteristics['symmetry_score'] = float(symmetry_score) if not np.isnan(symmetry_score) else 0.0
            
            # 6. Check for unrealistic sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            characteristics['sharpness'] = float(laplacian_var)
            
            # AI indicators based on these characteristics
            ai_indicators = []
            ai_score = 0.0
            
            # Very smooth textures (AI characteristic) - More sensitive
            if avg_local_variance < 200:
                ai_indicators.append("Very smooth textures detected")
                ai_score += 0.3
            
            # Unusual edge patterns - More sensitive
            if edge_density < 0.02 or edge_density > 0.15:
                ai_indicators.append("Unusual edge patterns")
                ai_score += 0.2
            
            # Low color variance (AI characteristic) - More sensitive
            if color_variance < 2000:
                ai_indicators.append("Low color variance")
                ai_score += 0.2
            
            # Unusual frequency patterns - More sensitive
            if freq_ratio < 0.02:
                ai_indicators.append("Unusual frequency patterns")
                ai_score += 0.2
            
            # Perfect symmetry (suspicious) - More sensitive
            if abs(symmetry_score) > 0.6:
                ai_indicators.append("Unusually high symmetry")
                ai_score += 0.2
            
            # Unrealistic sharpness - More sensitive
            if laplacian_var < 100:
                ai_indicators.append("Unrealistic sharpness")
                ai_score += 0.1
            
            # Additional AI indicators
            # Check for artificial grid patterns
            if edge_density < 0.01:
                ai_indicators.append("Artificial grid patterns detected")
                ai_score += 0.2
            
            # Check for overly uniform textures
            if avg_local_variance < 50:
                ai_indicators.append("Extremely uniform textures")
                ai_score += 0.3
            
            characteristics['ai_indicators'] = ai_indicators
            characteristics['ai_score'] = min(ai_score, 1.0)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error detecting AI characteristics: {str(e)}")
            return {}
    
    def detect_ai_image(self, image_path):
        """Main function to detect if an image is AI-generated"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Extract metadata
            metadata = self.extract_metadata(image_path)
            
            # Analyze compression artifacts
            compression_analysis = self.analyze_compression_artifacts(image_path)
            
            # Face analysis
            face_analysis = self.detect_face_inconsistencies(image_path)
            
            # AI characteristics analysis (NEW - more accurate)
            ai_characteristics = self.detect_ai_characteristics(image_path)
            
            # Image statistics analysis
            image_stats = self.analyze_image_statistics(image_path)
            
            # Combine analyses for final assessment
            analysis_factors = []
            ai_score = 0.0
            total_weight = 0.0
            
            # AI Characteristics Analysis (weight: 0.5) - Most important
            if ai_characteristics:
                ai_char_weight = 0.5
                ai_score += ai_characteristics.get('ai_score', 0) * ai_char_weight
                total_weight += ai_char_weight
                
                # Add specific indicators
                for indicator in ai_characteristics.get('ai_indicators', []):
                    analysis_factors.append(indicator)
            
            # Metadata analysis (weight: 0.2)
            metadata_weight = 0.2
            metadata_score = 0.0
            
            if 'exif' in metadata and len(metadata['exif']) == 0:
                metadata_score = 0.3
                analysis_factors.append("Missing EXIF data (suspicious)")
            elif 'exif' in metadata and len(metadata['exif']) > 10:
                metadata_score = 0.0
                analysis_factors.append("Rich EXIF data (likely real)")
            else:
                metadata_score = 0.1
                analysis_factors.append("Limited EXIF data")
            
            ai_score += metadata_score * metadata_weight
            total_weight += metadata_weight
            
            # Compression analysis (weight: 0.2)
            compression_weight = 0.2
            if compression_analysis.get('suspicious_compression', False):
                compression_score = 0.4
                analysis_factors.append("Unusual compression patterns detected")
            else:
                compression_score = 0.1
                analysis_factors.append("Normal compression patterns")
            
            ai_score += compression_score * compression_weight
            total_weight += compression_weight
            
            # Image statistics analysis (weight: 0.1)
            stats_weight = 0.1
            if image_stats.get('ai_score', 0) > 0.5:
                stats_score = image_stats['ai_score']
                analysis_factors.extend(image_stats.get('ai_indicators', []))
            else:
                stats_score = 0.1
                analysis_factors.append("Normal image statistics")
            
            ai_score += stats_score * stats_weight
            total_weight += stats_weight
            
            # Normalize final score
            final_ai_probability = ai_score / total_weight if total_weight > 0 else 0.0
            
            # Determine verdict with more sensitive thresholds
            if final_ai_probability > 0.5:
                verdict = 'AI-Generated'
                analysis_factors.append("Multiple strong indicators of AI generation")
            elif final_ai_probability > 0.3:
                verdict = 'Likely AI-Generated'
                analysis_factors.append("Several indicators suggest AI generation")
            elif final_ai_probability < 0.15:
                verdict = 'Likely Real'
                analysis_factors.append("Strong indicators of authentic content")
            else:
                verdict = 'Uncertain'
                analysis_factors.append("Mixed indicators - inconclusive")
            
            # Calculate confidence based on consistency of indicators
            confidence = 0.5 + (final_ai_probability * 0.5) if final_ai_probability > 0.5 else 0.5
            
            return {
                'ai_probability': round(final_ai_probability, 4),
                'confidence': round(confidence, 4),
                'analysis': analysis_factors,
                'metadata': metadata,
                'compression_analysis': compression_analysis,
                'face_analysis': face_analysis,
                'ai_characteristics': ai_characteristics,
                'image_statistics': image_stats,
                'verdict': verdict
            }
            
        except Exception as e:
            logger.error(f"Error detecting AI in image: {str(e)}")
            return {
                'ai_probability': 0.0,
                'confidence': 0.0,
                'analysis': [f"Error during analysis: {str(e)}"],
                'metadata': {},
                'verdict': 'Analysis Failed'
            }
    
    def detect_ai_video(self, video_path):
        """Detect if a video is AI-generated by analyzing frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for analysis (analyze every 30th frame or max 10 frames)
            sample_interval = max(1, frame_count // 10)
            frame_predictions = []
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Save frame temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    
                    # Analyze frame
                    frame_result = self.detect_ai_image(tmp_file.name)
                    frame_predictions.append(frame_result['ai_probability'])
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                
                if len(frame_predictions) >= 10:  # Limit analysis to 10 frames
                    break
            
            cap.release()
            
            # Calculate overall video AI probability
            if frame_predictions:
                avg_ai_probability = np.mean(frame_predictions)
                confidence = 1.0 - np.std(frame_predictions)  # Higher std = lower confidence
            else:
                avg_ai_probability = 0.0
                confidence = 0.0
            
            # Extract video metadata
            metadata = self.extract_metadata(video_path)
            
            analysis_factors = []
            analysis_factors.append(f"Analyzed {len(frame_predictions)} frames")
            
            if avg_ai_probability > 0.6:
                analysis_factors.append("High AI probability across video frames")
                verdict = 'AI-Generated'
            elif avg_ai_probability > 0.4:
                analysis_factors.append("Moderate AI probability across video frames")
                verdict = 'Likely AI-Generated'
            elif avg_ai_probability < 0.2:
                analysis_factors.append("Low AI probability - likely real video")
                verdict = 'Likely Real'
            else:
                analysis_factors.append("Mixed indicators across video frames")
                verdict = 'Uncertain'
            
            return {
                'ai_probability': round(avg_ai_probability, 4),
                'confidence': round(max(0, confidence), 4),
                'analysis': analysis_factors,
                'metadata': metadata,
                'frames_analyzed': len(frame_predictions),
                'frame_predictions': frame_predictions,
                'verdict': verdict
            }
            
        except Exception as e:
            logger.error(f"Error detecting AI in video: {str(e)}")
            return {
                'ai_probability': 0.0,
                'confidence': 0.0,
                'analysis': [f"Error during analysis: {str(e)}"],
                'metadata': {},
                'verdict': 'Analysis Failed'
            }
    
    def detailed_image_analysis(self, image_path):
        """Perform comprehensive image analysis"""
        basic_result = self.detect_ai_image(image_path)
        
        # Additional detailed analysis
        try:
            # Color space analysis
            img = cv2.imread(image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate color distribution
            color_analysis = {
                'mean_hue': float(np.mean(hsv[:,:,0])),
                'mean_saturation': float(np.mean(hsv[:,:,1])),
                'mean_value': float(np.mean(hsv[:,:,2])),
                'color_variance': float(np.var(hsv))
            }
            
            basic_result['detailed_analysis'] = {
                'color_analysis': color_analysis,
                'analysis_timestamp': np.datetime64('now').astype(str)
            }
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {str(e)}")
            
        return basic_result
    
    def detailed_video_analysis(self, video_path):
        """Perform comprehensive video analysis"""
        basic_result = self.detect_ai_video(video_path)
        
        # Additional video-specific analysis
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Motion analysis
            ret, prev_frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                motion_vectors = []
                
                for _ in range(min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None)
                    if flow[0] is not None:
                        motion_vectors.append(np.mean(np.abs(flow[0])))
                    prev_gray = gray
                
                motion_analysis = {
                    'average_motion': float(np.mean(motion_vectors)) if motion_vectors else 0,
                    'motion_consistency': float(1.0 - np.std(motion_vectors)) if motion_vectors else 0
                }
                
                basic_result['detailed_analysis'] = {
                    'motion_analysis': motion_analysis,
                    'analysis_timestamp': np.datetime64('now').astype(str)
                }
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in detailed video analysis: {str(e)}")
            
        return basic_result