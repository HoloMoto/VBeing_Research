/**
 * Pneuma - 3D Character Implementation
 * 
 * This file implements a 3D VRM character that represents
 * the Pneuma personal assistant using Three.js and the VRM library.
 */

// Character class for VRM model implementation
class Character {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById('character-canvas');
        this.isInitialized = false;
        this.isAnimating = false;

        // Three.js properties
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;
        this.mixer = null;

        // VRM model and animation properties
        this.vrm = null;
        this.currentAnimation = null;
        this.animationState = 'idle';

        // Default VRM model path
        this.modelPath = '/static/models/pneuma/model.vrm';

        // Animation parameters
        this.blinkInterval = null;
        this.mouthOpenValue = 0;
        this.isBlinking = false;
    }

    // Initialize the 3D character with Three.js and VRM
    async initialize() {
        console.log('Initializing 3D VRM character');

        // Check if canvas and container exist
        if (!this.canvas) {
            console.error('Character canvas element not found');
            return false;
        }

        if (!this.container) {
            console.error('Character container element not found');
            return false;
        }

        // Show the canvas
        this.canvas.style.display = 'block';

        // Set up the scene
        this.scene = new THREE.Scene();

        // Set up the camera
        const width = this.container.clientWidth;
        const height = 400; // Fixed height for the character
        this.camera = new THREE.PerspectiveCamera(30, width / height, 0.1, 20);
        this.camera.position.set(0, 1.3, 1.5);
        this.camera.lookAt(0, 1.3, 0);

        // Set up the renderer
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

        try {
            // Load the VRM model
            await this.loadVRMModel(this.modelPath);

            // Start the animation loop
            this.animate();

            // Start blinking
            this.startBlinking();

            this.isInitialized = true;
            console.log('VRM character initialized successfully');

            // Set initial state to idle
            this.idle();

            return true;
        } catch (error) {
            console.error('Failed to initialize VRM character:', error);

            // Show a fallback message if the model couldn't be loaded
            this.showFallbackMessage('VRMモデルの読み込みに失敗しました。モデルファイルを確認してください。');

            return false;
        }
    }

    // Load a VRM model from the given path
    async loadVRMModel(modelPath) {
        return new Promise((resolve, reject) => {
            // First, try to ensure GLTFLoader is attached to THREE namespace
            if (typeof GLTFLoader === 'function' && typeof THREE.GLTFLoader === 'undefined') {
                console.log("Attaching GLTFLoader to THREE namespace");
                THREE.GLTFLoader = GLTFLoader;
            }

            // Create a loader - handle both cases where GLTFLoader might be available
            let loader = null;

            // Try different ways to access GLTFLoader
            if (typeof THREE.GLTFLoader === 'function') {
                loader = new THREE.GLTFLoader();
                console.log("Using THREE.GLTFLoader");
            } else if (typeof GLTFLoader === 'function') {
                loader = new GLTFLoader();
                console.log("Using global GLTFLoader");
            } else {
                console.error("GLTFLoader not found in any namespace");
                // Try to dynamically load GLTFLoader
                const script = document.createElement('script');
                script.src = "https://unpkg.com/three@0.149.0/examples/js/loaders/GLTFLoader.js";
                script.onload = () => {
                    if (typeof GLTFLoader === 'function') {
                        THREE.GLTFLoader = GLTFLoader;
                        const loader = new THREE.GLTFLoader();
                        this.continueLoading(loader, modelPath, resolve, reject);
                    } else {
                        reject(new Error('Failed to load GLTFLoader dynamically'));
                    }
                };
                script.onerror = () => {
                    reject(new Error('Failed to load GLTFLoader script'));
                };
                document.head.appendChild(script);
                return;
            }

            if (!loader) {
                return reject(new Error('GLTFLoader not found. Make sure Three.js and GLTFLoader are properly loaded.'));
            }

            this.continueLoading(loader, modelPath, resolve, reject);
        });
    }

    // Helper method to continue loading once we have a loader
    continueLoading(loader, modelPath, resolve, reject) {
        // Load the model
        loader.load(
            modelPath,
            (gltf) => {
                // Convert the GLTF to VRM
                THREE.VRM.from(gltf).then(vrm => {
                    // Remove any existing model
                    if (this.vrm) {
                        this.scene.remove(this.vrm.scene);
                    }

                    // Add the new model to the scene
                    this.vrm = vrm;
                    this.scene.add(this.vrm.scene);

                    // Adjust model position
                    this.vrm.scene.position.set(0, 0, 0);

                    // Look at the camera
                    this.vrm.lookAt(new THREE.Vector3(0, 1.3, 1.5));

                    resolve(vrm);
                });
            },
            (progress) => {
                console.log('Loading VRM model:', (progress.loaded / progress.total * 100).toFixed(2) + '%');
            },
            (error) => {
                console.error('Error loading VRM model:', error);
                reject(error);
            }
        );
    }

    // Animation loop
    animate() {
        if (!this.isInitialized) return;

        requestAnimationFrame(this.animate.bind(this));

        const delta = this.clock.getDelta();

        // Update VRM animations
        if (this.vrm) {
            // Update mouth movement for speaking
            if (this.animationState === 'speaking') {
                // Oscillate mouth open value for speaking effect
                this.mouthOpenValue = 0.3 + 0.2 * Math.sin(Date.now() * 0.01);
                this.vrm.blendShapeProxy.setValue('a', this.mouthOpenValue);
            } else {
                // Close mouth when not speaking
                this.mouthOpenValue = 0;
                this.vrm.blendShapeProxy.setValue('a', 0);
            }

            // Update blinking
            if (this.isBlinking) {
                this.vrm.blendShapeProxy.setValue('blink', 1);
            } else {
                this.vrm.blendShapeProxy.setValue('blink', 0);
            }

            // Update VRM
            this.vrm.update(delta);
        }

        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }

    // Start blinking at random intervals
    startBlinking() {
        // Clear any existing interval
        if (this.blinkInterval) {
            clearInterval(this.blinkInterval);
        }

        // Set up new blinking interval
        this.blinkInterval = setInterval(() => {
            // Start blinking
            this.isBlinking = true;

            // Stop blinking after 150ms
            setTimeout(() => {
                this.isBlinking = false;
            }, 150);
        }, 3000 + Math.random() * 5000); // Random interval between 3-8 seconds
    }

    // Play a speaking animation when audio is playing
    speak(duration) {
        if (!this.isInitialized) {
            console.warn('Character not initialized');
            return;
        }

        console.log(`Character speak animation - duration: ${duration}ms`);
        this.animationState = 'speaking';
        this.isAnimating = true;

        // Reset to idle after the duration
        setTimeout(() => {
            this.idle();
        }, duration);
    }

    // Play an idle animation
    idle() {
        if (!this.isInitialized) {
            console.warn('Character not initialized');
            return;
        }

        console.log('Character idle animation');
        this.animationState = 'idle';
        this.isAnimating = false;
    }

    // React to user input with appropriate expression
    react(emotion) {
        if (!this.isInitialized || !this.vrm) {
            console.warn('Character not initialized');
            return;
        }

        console.log(`Character reaction - emotion: ${emotion}`);

        // Reset all expressions
        this.vrm.blendShapeProxy.setValue('happy', 0);
        this.vrm.blendShapeProxy.setValue('angry', 0);
        this.vrm.blendShapeProxy.setValue('sad', 0);
        this.vrm.blendShapeProxy.setValue('surprised', 0);

        // Set the appropriate expression
        switch (emotion) {
            case 'happy':
                this.vrm.blendShapeProxy.setValue('happy', 1);
                break;
            case 'angry':
                this.vrm.blendShapeProxy.setValue('angry', 1);
                break;
            case 'sad':
                this.vrm.blendShapeProxy.setValue('sad', 1);
                break;
            case 'surprised':
                this.vrm.blendShapeProxy.setValue('surprised', 1);
                break;
            default:
                // No expression change for unknown emotions
                break;
        }

        // Reset expression after 2 seconds
        setTimeout(() => {
            this.vrm.blendShapeProxy.setValue('happy', 0);
            this.vrm.blendShapeProxy.setValue('angry', 0);
            this.vrm.blendShapeProxy.setValue('sad', 0);
            this.vrm.blendShapeProxy.setValue('surprised', 0);
        }, 2000);
    }

    // Show a fallback message when the model can't be loaded
    showFallbackMessage(message) {
        // Hide the canvas
        this.canvas.style.display = 'none';

        // Show a message in the container
        this.container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #666;">
                <p>${message}</p>
                <p>static/models/pneuma/model.vrm を配置してください。</p>
            </div>
        `;
    }

    // Handle window resize
    resize() {
        if (!this.isInitialized) return;

        const width = this.container.clientWidth;
        const height = 400; // Fixed height

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }
}

// Global character instance
let character = null;

// Function to initialize the character
function initializeCharacter() {
    // Check if WebGL is available
    const canvas = document.createElement('canvas');
    const hasWebGL = !!(window.WebGLRenderingContext && 
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));

    if (!hasWebGL) {
        console.warn('WebGL not supported - 3D character cannot be initialized');
        return false;
    }

    // Check if the container and canvas elements exist
    const container = document.getElementById('character-container');
    const characterCanvas = document.getElementById('character-canvas');

    if (!container || !characterCanvas) {
        console.error('Character container or canvas element not found in the DOM');
        return false;
    }

    // Create and initialize the character
    character = new Character('character-container');

    // Initialize the character and set up event listeners
    character.initialize().then(success => {
        if (success) {
            console.log('VRM character initialized successfully');

            // Update the global window.character reference immediately after successful initialization
            window.character = character;

            // Set up window resize handler
            window.addEventListener('resize', () => {
                character.resize();
            });

            // Connect character to audio playback events
            document.addEventListener('audio-play-start', (event) => {
                const duration = event.detail.duration || 5000;
                character.speak(duration);
            });

            document.addEventListener('audio-play-end', () => {
                character.idle();
            });
        } else {
            console.warn('Failed to initialize VRM character');
        }
    }).catch(error => {
        console.error('Error during character initialization:', error);
    });

    return true;
}

// Initialize the character when the DOM is loaded and all scripts are ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for the window load event to ensure all scripts are loaded
    window.addEventListener('load', () => {
        // Add a small delay to ensure all scripts are initialized
        setTimeout(() => {
            // Check if THREE is available
            if (typeof THREE === 'undefined') {
                console.error('THREE.js library not loaded properly');

                // Try to load THREE.js dynamically
                const threeScript = document.createElement('script');
                threeScript.src = "https://unpkg.com/three@0.149.0/build/three.min.js";
                threeScript.onload = () => {
                    console.log("THREE.js loaded dynamically");
                    checkAndLoadGLTFLoader();
                };
                document.head.appendChild(threeScript);
                return;
            }

            checkAndLoadGLTFLoader();
        }, 1000);
    });
});

// Helper function to check and load GLTFLoader if needed
function checkAndLoadGLTFLoader() {
    // If GLTFLoader is not available in either namespace, try to load it
    if (typeof THREE.GLTFLoader === 'undefined' && typeof GLTFLoader === 'undefined') {
        console.log('GLTFLoader not found, attempting to load it dynamically');

        const gltfScript = document.createElement('script');
        gltfScript.src = "https://unpkg.com/three@0.149.0/examples/js/loaders/GLTFLoader.js";
        gltfScript.onload = () => {
            console.log("GLTFLoader loaded dynamically");

            // Attach GLTFLoader to THREE namespace if it's available globally
            if (typeof GLTFLoader === 'function' && typeof THREE.GLTFLoader === 'undefined') {
                console.log("Attaching GLTFLoader to THREE namespace");
                THREE.GLTFLoader = GLTFLoader;
            }

            // Check if VRM library is loaded
            checkAndLoadVRM();
        };
        document.head.appendChild(gltfScript);
    } else {
        // GLTFLoader is available, check VRM
        checkAndLoadVRM();
    }
}

// Helper function to check and load VRM if needed
function checkAndLoadVRM() {
    // If VRM is not available, try to load it
    if (typeof THREE.VRM === 'undefined') {
        console.log('VRM library not found, attempting to load it dynamically');

        const vrmScript = document.createElement('script');
        vrmScript.src = "https://unpkg.com/@pixiv/three-vrm@0.6.7/lib/three-vrm.min.js";
        vrmScript.onload = () => {
            console.log("VRM library loaded dynamically");
            initializeCharacterWhenReady();
        };
        document.head.appendChild(vrmScript);
    } else {
        // VRM is available, initialize character
        initializeCharacterWhenReady();
    }
}

// Initialize character when all dependencies are ready
function initializeCharacterWhenReady() {
    // Initialize the character
    const success = initializeCharacter();

    // Update the global window.character reference only if initialization was successful
    if (success && character) {
        window.character = character;
        console.log('Global character reference updated');
    }
}

// Export for use in other scripts
window.GeminiCharacter = {
    Character,
    initializeCharacter,
    getInstance: () => character
};

// Also make the character instance available directly on window for compatibility with existing code
window.character = character;
