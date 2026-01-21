#!/usr/bin/env python3
"""
Minimal diagnostic_loader module for model loading and diagnostics
"""
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load models and pattern files from models directory"""
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path('models')
        self.models = {}
        self.patterns = {}
    
    def load_all(self):
        """Load all available models and patterns"""
        return {
            'models_dir': str(self.models_dir),
            'status': 'Models will be loaded on-demand'
        }
    
    def get_model(self, model_name: str):
        """Load and return a specific model"""
        try:
            from tensorflow import keras
            
            # Try .keras format first, then .h5
            keras_path = self.models_dir / f"{model_name}.keras"
            h5_path = self.models_dir / f"{model_name}.h5"
            
            if keras_path.exists():
                logger.info(f"Loading {model_name}.keras...")
                model = keras.models.load_model(str(keras_path))
                logger.info(f"Successfully loaded {model_name}.keras")
                return model
            elif h5_path.exists():
                logger.info(f"Loading {model_name}.h5...")
                model = keras.models.load_model(str(h5_path))
                logger.info(f"Successfully loaded {model_name}.h5")
                return model
            else:
                logger.warning(f"Model {model_name} not found in {self.models_dir}")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_pattern(self, pattern_name: str):
        """Load and return a pattern file"""
        try:
            pattern_path = self.models_dir / f"{pattern_name}.npz"
            
            if pattern_path.exists():
                logger.debug(f"Loading pattern {pattern_name}")
                data = np.load(str(pattern_path), allow_pickle=True)
                return data
            else:
                logger.warning(f"Pattern {pattern_name} not found at {pattern_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading pattern {pattern_name}: {e}")
            return None


class SystemDiagnostics:
    """System diagnostics for model validation"""
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path('models')
        self.diagnostics_report = {}
    
    def run_full_diagnosis(self, save_to: str = None):
        """Run full system diagnosis"""
        try:
            report = {
                'timestamp': str(Path.cwd()),
                'models_dir': str(self.models_dir),
                'models_found': [],
                'patterns_found': [],
                'status': 'OK'
            }
            
            # Check for models
            for model_file in self.models_dir.glob('*.h5'):
                report['models_found'].append(model_file.name)
            
            for model_file in self.models_dir.glob('*.keras'):
                report['models_found'].append(model_file.name)
            
            # Check for patterns
            for pattern_file in self.models_dir.glob('*.npz'):
                report['patterns_found'].append(pattern_file.name)
            
            self.diagnostics_report = report
            
            # Save if requested
            if save_to:
                try:
                    with open(save_to, 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Diagnostics report saved to {save_to}")
                except Exception as e:
                    logger.warning(f"Could not save diagnostics report: {e}")
            
            return report
        
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
