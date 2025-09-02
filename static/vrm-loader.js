/**
 * VBeing Research - Enhanced VRM Model Loader
 * 
 * This file implements an improved VRM character loader based on the ChatVRMG approach
 * but adapted for the VBeing Research project.
 */

// Model class for VRM implementation
class VRMModel {
    constructor(container) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        this.canvas = document.getElementById('character-canvas');
        this.isInitialized = false;

        // Three.js properties
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;

        // VRM model properties
        this.vrm = null;
        this.lookAtTarget = null;

        // Animation properties
        this.blinkController = null;
        this.expressionController = null;
        this.currentState = 'idle';

        // Default model path
        this.modelPath = '/static/models/pneuma/model.vrm';

        // Logging function for debugging
        this.log = (message, type = 'info') => {
            const prefix = '[VRM Loader]';
            switch(type) {
                case 'error': console.error(`${prefix} ${message}`); break;
                case 'warn': console.warn(`${prefix} ${message}`); break;
                default: console.log(`${prefix} ${message}`);
            }

            // Also log to server if needed
            this.logToServer(type, message);
        };
    }

    // Log to server for debugging
    logToServer(eventType, message, details = {}) {
        fetch('/api/log_character_event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                event_type: eventType,
                message: message,
                details: details
            })
        }).catch(error => {
            console.error('Error sending log to server:', error);
        });
    }

    // Initialize the 3D scene
    async initialize() {
        this.log('Initializing VRM character');

        if (!this.canvas || !this.container) {
            this.log('Canvas or container element not found', 'error');
            return false;
        }

        // Show the canvas
        this.canvas.style.display = 'block';

        try {
            // Set up Three.js scene
            this.setupScene();

            // Load the VRM model
            await this.loadVRMModel(this.modelPath);

            // Start animation loop
            this.startAnimationLoop();

            // Initialize controllers
            this.initializeControllers();

            this.isInitialized = true;
            this.log('VRM character initialized successfully');

            return true;
        } catch (error) {
            this.log(`Failed to initialize VRM character: ${error.message}`, 'error');
            this.showFallbackMessage('VRMモデルの読み込みに失敗しました。モデルファイルを確認してください。');
            return false;
        }
    }

    // Set up the Three.js scene
    setupScene() {
        // Create scene
        this.scene = new THREE.Scene();

        // Set up camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;
        this.camera = new THREE.PerspectiveCamera(30, width / height, 0.1, 20);
        this.camera.position.set(0, 1.3, 1.5);
        this.camera.lookAt(0, 1.3, 0);

        // Set up renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: true,
            antialias: true
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Set up lighting
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Set up clock for animations
        this.clock = new THREE.Clock();
        this.clock.start();

        // Create look-at target
        this.lookAtTarget = new THREE.Object3D();
        this.camera.add(this.lookAtTarget);

        this.log('Three.js scene set up successfully');
    }

    // Load the VRM model
    async loadVRMModel(modelPath) {
        this.log(`Loading VRM model from: ${modelPath}`);

        // Show loading progress
        const loadingElement = document.createElement('div');
        loadingElement.style.position = 'absolute';
        loadingElement.style.top = '50%';
        loadingElement.style.left = '50%';
        loadingElement.style.transform = 'translate(-50%, -50%)';
        loadingElement.style.background = 'rgba(255, 255, 255, 0.8)';
        loadingElement.style.padding = '20px';
        loadingElement.style.borderRadius = '10px';
        loadingElement.style.textAlign = 'center';
        loadingElement.innerHTML = `
            <p>VRMモデルを読み込み中...</p>
            <div style="width: 100%; height: 10px; background: #eee; border-radius: 5px; overflow: hidden; margin: 10px 0;">
                <div id="vrm-progress-bar" style="width: 0%; height: 100%; background: #4285f4; transition: width 0.3s;"></div>
            </div>
            <p id="vrm-progress-text">0%</p>
        `;
        this.container.appendChild(loadingElement);

        try {
            // Check if GLTFLoader is available globally or on THREE
            const GLTFLoaderClass = window.GLTFLoader || THREE.GLTFLoader;

            if (!GLTFLoaderClass) {
                throw new Error("GLTFLoader not found. Make sure Three.js and GLTFLoader are properly loaded.");
            }

            // Create GLTFLoader
            const loader = new GLTFLoaderClass();

            // Check if VRMLoaderPlugin is available
            const VRMLoaderPluginClass = window.VRMLoaderPlugin || (THREE.VRM && THREE.VRM.VRMLoaderPlugin);

            if (VRMLoaderPluginClass) {
                // Register VRM plugin
                loader.register((parser) => {
                    return new VRMLoaderPluginClass(parser);
                });
            } else {
                console.warn("VRMLoaderPlugin not found, trying to load VRM without plugin");
            }

            // Load the model with progress tracking
            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    modelPath,
                    resolve,
                    (progress) => {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        const progressBar = document.getElementById('vrm-progress-bar');
                        const progressText = document.getElementById('vrm-progress-text');
                        if (progressBar) progressBar.style.width = `${percent}%`;
                        if (progressText) progressText.textContent = `${percent}%`;
                    },
                    reject
                );
            });

            // Get VRM from gltf.userData or use VRM.from
            let vrm;
            try {
                if (gltf.userData && gltf.userData.vrm) {
                    console.log('VRM found in gltf.userData.vrm');
                    vrm = gltf.userData.vrm;
                } else if (window.VRM && window.VRM.from) {
                    console.log('Using window.VRM.from');
                    // Use global VRM.from if available
                    const result = window.VRM.from(gltf);
                    // Handle both Promise and direct return cases
                    vrm = result instanceof Promise ? await result : result;
                } else if (THREE.VRM && THREE.VRM.from) {
                    console.log('Using THREE.VRM.from');
                    // Fallback to THREE.VRM.from if available
                    const result = THREE.VRM.from(gltf);
                    // Handle both Promise and direct return cases
                    vrm = result instanceof Promise ? await result : result;
                } else {
                    throw new Error("Could not extract VRM from loaded model - no VRM.from method available");
                }

                if (!vrm) {
                    throw new Error("VRM extraction returned null or undefined");
                }

                console.log('VRM extracted successfully:', vrm);
            } catch (error) {
                console.error('Error extracting VRM from model:', error);
                throw new Error(`Failed to extract VRM: ${error.message}`);
            }

            this.vrm = vrm;

            // Add to scene
            this.scene.add(this.vrm.scene);

            // Set up VRM
            if (this.vrm.lookAt) {
                this.vrm.lookAt.target = this.lookAtTarget;
            }

            // Remove loading element
            loadingElement.remove();

            this.log('VRM model loaded successfully');
            return this.vrm;

        } catch (error) {
            // Remove loading element
            loadingElement.remove();

            this.log(`Error loading VRM model: ${error.message}`, 'error');
            throw error;
        }
    }

    // Initialize expression and blink controllers
    initializeControllers() {
        if (!this.vrm) return;

        // Initialize blink controller
        this.blinkController = {
            isBlinking: false,
            interval: null,

            start: () => {
                if (this.blinkController.interval) {
                    clearInterval(this.blinkController.interval);
                }

                this.blinkController.interval = setInterval(() => {
                    // Start blinking
                    this.blinkController.isBlinking = true;
                    this.vrm.expressionManager?.setValue('blink', 1);

                    // Stop blinking after 150ms
                    setTimeout(() => {
                        this.blinkController.isBlinking = false;
                        this.vrm.expressionManager?.setValue('blink', 0);
                    }, 150);
                }, 3000 + Math.random() * 5000); // Random interval between 3-8 seconds
            },

            stop: () => {
                if (this.blinkController.interval) {
                    clearInterval(this.blinkController.interval);
                    this.blinkController.interval = null;
                }
                this.blinkController.isBlinking = false;
                this.vrm.expressionManager?.setValue('blink', 0);
            }
        };

        // Initialize expression controller
        this.expressionController = {
            currentExpression: 'neutral',

            setExpression: (expression, value = 1.0) => {
                if (!this.vrm.expressionManager) return;

                // Reset current expression
                if (this.expressionController.currentExpression !== 'neutral') {
                    this.vrm.expressionManager.setValue(this.expressionController.currentExpression, 0);
                }

                // Set new expression
                this.expressionController.currentExpression = expression;
                this.vrm.expressionManager.setValue(expression, value);
            },

            lipSync: (value) => {
                if (!this.vrm.expressionManager) return;

                // Set mouth open value
                this.vrm.expressionManager.setValue('aa', value);
            }
        };

        // Start blinking
        this.blinkController.start();

        this.log('Controllers initialized successfully');
    }

    // Start the animation loop
    startAnimationLoop() {
        const animate = () => {
            if (!this.isInitialized) return;

            requestAnimationFrame(animate);

            const delta = this.clock.getDelta();

            // Update VRM
            if (this.vrm) {
                // Update lip sync if speaking
                if (this.currentState === 'speaking') {
                    const mouthOpenValue = 0.3 + 0.2 * Math.sin(Date.now() * 0.01);
                    this.expressionController?.lipSync(mouthOpenValue);
                } else {
                    this.expressionController?.lipSync(0);
                }

                // Update VRM
                this.vrm.update(delta);
            }

            // Render scene
            this.renderer.render(this.scene, this.camera);
        };

        animate();
        this.log('Animation loop started');
    }

    // Play speaking animation
    speak(duration) {
        if (!this.isInitialized) {
            this.log('Cannot speak - character not initialized', 'warn');
            return;
        }

        this.log(`Speaking animation started (duration: ${duration}ms)`);
        this.currentState = 'speaking';

        // Reset to idle after duration
        setTimeout(() => {
            this.idle();
        }, duration);
    }

    // Set idle animation
    idle() {
        if (!this.isInitialized) {
            this.log('Cannot set idle - character not initialized', 'warn');
            return;
        }

        this.log('Idle animation set');
        this.currentState = 'idle';
    }

    // Set expression
    setExpression(expression) {
        if (!this.isInitialized || !this.expressionController) {
            this.log('Cannot set expression - character not initialized', 'warn');
            return;
        }

        this.log(`Setting expression: ${expression}`);
        this.expressionController.setExpression(expression);

        // Reset expression after 2 seconds
        setTimeout(() => {
            this.expressionController.setExpression('neutral');
        }, 2000);
    }

    // Show fallback message when model loading fails
    showFallbackMessage(message) {
        this.canvas.style.display = 'none';

        this.container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #666;">
                <p>${message}</p>
                <p>static/models/pneuma/model.vrm を配置してください。</p>
                <button id="retry-vrm-load" style="padding: 8px 16px; background: #4285f4; color: white; border: none; border-radius: 4px; margin-top: 10px; cursor: pointer;">
                    再試行
                </button>
            </div>
        `;

        // Add retry button functionality
        setTimeout(() => {
            const retryButton = document.getElementById('retry-vrm-load');
            if (retryButton) {
                retryButton.addEventListener('click', () => {
                    window.location.reload();
                });
            }
        }, 100);
    }

    // Handle window resize
    resize() {
        if (!this.isInitialized) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }
}

// Global instance
let vrmCharacter = null;

// Initialize the character
function initializeVRMCharacter(containerId = 'character-container') {
    // Check WebGL support
    const canvas = document.createElement('canvas');
    const hasWebGL = !!(window.WebGLRenderingContext && 
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));

    if (!hasWebGL) {
        console.warn('WebGL not supported - 3D character cannot be initialized');
        return false;
    }

    // Create and initialize the character
    vrmCharacter = new VRMModel(containerId);

    // Initialize and set up event listeners
    vrmCharacter.initialize().then(success => {
        if (success) {
            console.log('VRM character initialized successfully');

            // Make globally available as both vrmCharacter and character for compatibility
            window.vrmCharacter = vrmCharacter;
            window.character = vrmCharacter;  // For compatibility with existing code

            // Set up window resize handler
            window.addEventListener('resize', () => {
                vrmCharacter.resize();
            });

            // Connect to audio playback events (for future compatibility)
            document.addEventListener('audio-play-start', (event) => {
                const duration = event.detail.duration || 5000;
                vrmCharacter.speak(duration);
            });

            document.addEventListener('audio-play-end', () => {
                vrmCharacter.idle();
            });
        }
    });

    return true;
}

// Add drag and drop support for VRM models
function setupDragAndDrop(container, vrmModel) {
    container.addEventListener('dragover', (event) => {
        event.preventDefault();
        container.style.border = '2px dashed #4285f4';
    });

    container.addEventListener('dragleave', () => {
        container.style.border = 'none';
    });

    container.addEventListener('drop', async (event) => {
        event.preventDefault();
        container.style.border = 'none';

        const files = event.dataTransfer?.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        const fileType = file.name.split('.').pop().toLowerCase();

        if (fileType === 'vrm') {
            try {
                // Create object URL from the file
                const blob = new Blob([file], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);

                // Remove current model
                if (vrmModel.vrm) {
                    vrmModel.scene.remove(vrmModel.vrm.scene);
                }

                // Load new model
                await vrmModel.loadVRMModel(url);

                // Initialize controllers for the new model
                vrmModel.initializeControllers();

                console.log(`Loaded VRM model from file: ${file.name}`);
            } catch (error) {
                console.error('Error loading dropped VRM file:', error);
                alert(`モデルの読み込みに失敗しました: ${error.message}`);
            }
        } else {
            alert('VRMファイル (.vrm) のみ対応しています。');
        }
    });
}

// Export for use in other scripts
window.VRMCharacter = {
    Model: VRMModel,
    initialize: initializeVRMCharacter,
    getInstance: () => vrmCharacter,
    setupDragAndDrop: setupDragAndDrop
};

// Also export as GeminiCharacter for compatibility with existing code
window.GeminiCharacter = window.VRMCharacter;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM content loaded, checking for dependencies...');

    // Log the current state of dependencies for debugging
    const logDependencyState = () => {
        console.log('Dependency state:');
        console.log('- THREE available:', typeof THREE !== 'undefined');
        console.log('- window.GLTFLoader available:', typeof window.GLTFLoader !== 'undefined');
        console.log('- THREE.GLTFLoader available:', typeof THREE !== 'undefined' && typeof THREE.GLTFLoader !== 'undefined');
        console.log('- window.VRM available:', typeof window.VRM !== 'undefined');
        console.log('- THREE.VRM available:', typeof THREE !== 'undefined' && typeof THREE.VRM !== 'undefined');
        console.log('- window.VRMLoaderPlugin available:', typeof window.VRMLoaderPlugin !== 'undefined');
    };

    // Log initial state
    logDependencyState();

    // Wait for Three.js and VRM to be loaded
    const checkDependencies = () => {
        // Check for THREE
        if (typeof THREE === 'undefined') {
            console.log('THREE not loaded yet, waiting...');
            setTimeout(checkDependencies, 100);
            return;
        }

        // Check for GLTFLoader (either global or on THREE)
        if (typeof window.GLTFLoader === 'undefined' && typeof THREE.GLTFLoader === 'undefined') {
            console.log('GLTFLoader not loaded yet, waiting...');
            setTimeout(checkDependencies, 100);
            return;
        }

        // Check for VRM (either global or on THREE)
        const vrmAvailable = (typeof window.VRM !== 'undefined') || 
                           (typeof THREE.VRM !== 'undefined') ||
                           (typeof window.VRMLoaderPlugin !== 'undefined');

        if (!vrmAvailable) {
            console.log('VRM libraries not loaded yet, waiting...');
            setTimeout(checkDependencies, 100);
            return;
        }

        console.log('All dependencies loaded, ensuring VRMLoaderPlugin is registered');

        // Ensure VRMLoaderPlugin is registered with GLTFLoader
        try {
            // Get the appropriate classes
            const GLTFLoaderClass = window.GLTFLoader || THREE.GLTFLoader;
            const VRMLoaderPluginClass = window.VRMLoaderPlugin || (THREE.VRM && THREE.VRM.VRMLoaderPlugin);

            if (GLTFLoaderClass && VRMLoaderPluginClass) {
                // Check if we need to manually register the plugin
                if (typeof GLTFLoaderClass.prototype.register === 'function') {
                    console.log('GLTFLoader has register method, VRMLoaderPlugin should work correctly');
                } else {
                    console.warn('GLTFLoader does not have register method, attempting to add it');
                    // This is a fallback for older versions or if the module loading didn't work correctly
                    GLTFLoaderClass.prototype.register = function(plugin) {
                        if (!this.plugins) this.plugins = [];
                        this.plugins.push(plugin);
                    };
                }
            }
        } catch (error) {
            console.error('Error ensuring VRMLoaderPlugin registration:', error);
        }

        console.log('Initializing VRM character');
        initializeVRMCharacter();

        // Set up drag and drop for the character container
        const container = document.getElementById('character-container');
        if (container && vrmCharacter) {
            setupDragAndDrop(container, vrmCharacter);
        }

        // Log final dependency state
        logDependencyState();
    };

    checkDependencies();
});
