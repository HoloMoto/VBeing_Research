/**
 * Pneuma - 3D Character Implementation using Babylon.js
 * 
 * This file implements a 3D VRM character that represents
 * the Pneuma personal assistant using Babylon.js.
 */

// Helper function to send logs to the server
function logToServer(eventType, message, details = {}) {
    console.log(`[Character ${eventType}] ${message}`);

    // Send the log to the server
    fetch('/api/log_character_event', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            event_type: eventType,
            message: message,
            details: details
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status !== 'success') {
            console.warn('Failed to log to server:', data.message);
        }
    })
    .catch(error => {
        console.error('Error sending log to server:', error);
    });
}

// Character class for VRM model implementation with Babylon.js
class BabylonCharacter {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.isInitialized = false;
        this.isAnimating = false;

        // Babylon.js properties
        this.engine = null;
        this.scene = null;
        this.camera = null;
        this.light = null;

        // VRM model and animation properties
        this.vrm = null;
        this.animationState = 'idle';

        // Debug options
        this.debug = options.debug || false;
        this.forceFallbackModel = options.forceFallbackModel || false;

        // Default VRM model path
        // First try to use the local model if available
        this.modelPath = '/static/models/pneuma/model.vrm';

        // Fallback to a known working VRM model for testing
        // Using the example from amayuski/babylon-vrm-0.0
        this.fallbackModelPath = 'https://raw.githubusercontent.com/vrm-c/UniVRM/master/Tests/Models/Alicia_vrm-0.51/AliciaSolid_vrm-0.51.vrm';

        // Animation parameters
        this.blinkInterval = null;
        this.isBlinking = false;
        this.mouthOpenValue = 0;

        // Log debug info if enabled
        if (this.debug) {
            console.log('BabylonCharacter initialized with debug options:', options);
        }
    }

    // Initialize the 3D character with Babylon.js
    async initialize() {
        console.log('Initializing 3D VRM character with Babylon.js');
        logToServer('init', 'Initializing 3D VRM character with Babylon.js');

        // Check if canvas exists
        if (!this.canvas) {
            console.error('Character canvas element not found');
            logToServer('error', 'Character canvas element not found');
            return false;
        }

        try {
            // Log Babylon.js initialization
            console.log('Creating Babylon.js engine');
            logToServer('init_step', 'Creating Babylon.js engine');

            // Create the Babylon engine
            this.engine = new BABYLON.Engine(this.canvas, true);
            console.log('Babylon.js engine created successfully');
            logToServer('init_step', 'Babylon.js engine created successfully');

            // Log scene creation
            console.log('Creating Babylon.js scene');
            logToServer('init_step', 'Creating Babylon.js scene');

            // Create a basic scene
            this.scene = new BABYLON.Scene(this.engine);
            this.scene.clearColor = new BABYLON.Color4(0, 0, 0, 0); // Transparent background
            console.log('Babylon.js scene created successfully');
            logToServer('init_step', 'Babylon.js scene created successfully');

            // Log camera creation
            console.log('Creating camera');
            logToServer('init_step', 'Creating camera');

            // Create camera
            this.camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 3, new BABYLON.Vector3(0, 1, 0), this.scene);
            this.camera.attachControl(this.canvas, true);
            this.camera.lowerRadiusLimit = 2;
            this.camera.upperRadiusLimit = 5;
            this.camera.wheelDeltaPercentage = 0.01;
            console.log('Camera created successfully');
            logToServer('init_step', 'Camera created successfully');

            // Log lights creation
            console.log('Creating lights');
            logToServer('init_step', 'Creating lights');

            // Create lights
            this.light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), this.scene);
            this.light.intensity = 0.7;

            // Add directional light for better shadows
            const directionalLight = new BABYLON.DirectionalLight("directionalLight", new BABYLON.Vector3(-1, -2, -1), this.scene);
            directionalLight.intensity = 0.5;
            console.log('Lights created successfully');
            logToServer('init_step', 'Lights created successfully');

            // Check if we should force using the fallback model
            if (this.forceFallbackModel) {
                console.log('Debug option forceFallbackModel is enabled, using fallback model directly');
                logToServer('init', 'Using fallback model directly (debug option enabled)');
                try {
                    // Check if the fallback model exists (for local fallback models)
                    if (this.fallbackModelPath.startsWith('/')) {
                        console.log('Checking if fallback VRM model exists:', this.fallbackModelPath);
                        logToServer('init', `Checking if fallback VRM model exists: ${this.fallbackModelPath}`);

                        const fallbackModelExists = await this.checkModelFileExists(this.fallbackModelPath);
                        if (!fallbackModelExists) {
                            console.warn(`Fallback VRM model file not found on server: ${this.fallbackModelPath}`);
                            logToServer('warning', `Fallback VRM model file not found on server: ${this.fallbackModelPath}`);
                            // Continue anyway since it might be a remote URL
                        }
                    }

                    await this.loadVRMModel(this.fallbackModelPath);
                    console.log('Fallback VRM model loaded successfully');
                    logToServer('success', 'Fallback VRM model loaded successfully');
                } catch (error) {
                    console.error('Failed to load fallback VRM model:', error);
                    logToServer('error', 'Failed to load fallback VRM model', {error: error.message, stack: error.stack});
                    this.showFallbackMessage('VRMモデルの読み込みに失敗しました。モデルファイルを確認してください。');
                    return false;
                }
            } else {
                try {
                    // First check if the primary model file exists
                    console.log('Checking if primary VRM model exists:', this.modelPath);
                    logToServer('init_step', `Checking if primary VRM model exists: ${this.modelPath}`);

                    const modelExists = await this.checkModelFileExists(this.modelPath);
                    if (modelExists) {
                        console.log(`Primary VRM model file found on server: ${this.modelPath}`);
                        logToServer('init_step', `Primary VRM model file found on server: ${this.modelPath}`);
                    } else {
                        console.warn(`Primary VRM model file not found on server: ${this.modelPath}`);
                        logToServer('warning', `Primary VRM model file not found on server: ${this.modelPath}`);
                        throw new Error(`Model file not found: ${this.modelPath}`);
                    }

                    // Then try to load the primary model
                    console.log('Attempting to load primary VRM model:', this.modelPath);
                    logToServer('init_step', `Attempting to load primary VRM model: ${this.modelPath}`);
                    await this.loadVRMModel(this.modelPath);
                    console.log('Primary VRM model loaded successfully');
                    logToServer('success', 'Primary VRM model loaded successfully');
                } catch (primaryError) {
                    console.warn('Failed to load primary VRM model:', primaryError);
                    logToServer('warning', 'Failed to load primary VRM model, trying fallback', {error: primaryError.message, stack: primaryError.stack});
                    console.log('Attempting to load fallback VRM model:', this.fallbackModelPath);

                    // Show a message about trying the fallback model
                    const loadingMessage = document.createElement('div');
                    loadingMessage.style.position = 'absolute';
                    loadingMessage.style.top = '50%';
                    loadingMessage.style.left = '50%';
                    loadingMessage.style.transform = 'translate(-50%, -50%)';
                    loadingMessage.style.textAlign = 'center';
                    loadingMessage.style.color = '#666';
                    loadingMessage.innerHTML = '<p>プライマリモデルの読み込みに失敗しました。フォールバックモデルを試しています...</p>';
                    this.canvas.parentNode.appendChild(loadingMessage);

                    try {
                        // Check if the fallback model exists (for local fallback models)
                        if (this.fallbackModelPath.startsWith('/')) {
                            console.log('Checking if fallback VRM model exists:', this.fallbackModelPath);
                            logToServer('init', `Checking if fallback VRM model exists: ${this.fallbackModelPath}`);

                            const fallbackModelExists = await this.checkModelFileExists(this.fallbackModelPath);
                            if (!fallbackModelExists) {
                                console.warn(`Fallback VRM model file not found on server: ${this.fallbackModelPath}`);
                                logToServer('warning', `Fallback VRM model file not found on server: ${this.fallbackModelPath}`);
                                // Continue anyway since it might be a remote URL
                            }
                        }

                        // Try to load the fallback model
                        logToServer('init', `Attempting to load fallback VRM model: ${this.fallbackModelPath}`);
                        await this.loadVRMModel(this.fallbackModelPath);
                        console.log('Fallback VRM model loaded successfully');
                        logToServer('success', 'Fallback VRM model loaded successfully');
                        loadingMessage.remove();
                    } catch (fallbackError) {
                        console.error('Failed to load fallback VRM model:', fallbackError);
                        logToServer('error', 'Failed to load fallback VRM model', {error: fallbackError.message, stack: fallbackError.stack});
                        loadingMessage.remove();
                        this.showFallbackMessage('VRMモデルの読み込みに失敗しました。モデルファイルを確認してください。');
                        return false;
                    }
                }
            }

            // Start the render loop
            this.startRenderLoop();

            // Start blinking
            this.startBlinking();

            this.isInitialized = true;
            console.log('Babylon VRM character initialized successfully');
            logToServer('success', 'Babylon VRM character initialized successfully');

            // Set initial state to idle
            console.log('Setting initial character state to idle');
            logToServer('init_step', 'Setting initial character state to idle');
            this.idle();
            console.log('Character initialization complete');
            logToServer('init_step', 'Character initialization complete');

            return true;
        } catch (error) {
            console.error('Failed to initialize Babylon VRM character:', error);
            logToServer('error', 'Failed to initialize Babylon VRM character', {error: error.message, stack: error.stack});
            this.showFallbackMessage('VRMモデルの読み込みに失敗しました。モデルファイルを確認してください。');
            return false;
        }
    }

    // Load a VRM model from the given path using babylon-vrm-loader
    async loadVRMModel(modelPath) {
        console.log(`Starting VRM model loading process for: ${modelPath}`);
        logToServer('model_loading', `Starting VRM model loading process for: ${modelPath}`);

        return new Promise((resolve, reject) => {
            // Set a timeout to prevent hanging if the model can't be loaded
            const timeoutMs = 30000; // 30 seconds timeout
            console.log(`Setting model loading timeout: ${timeoutMs}ms`);
            logToServer('model_loading', `Setting model loading timeout: ${timeoutMs}ms`);

            const timeoutId = setTimeout(() => {
                console.error(`Model loading timed out after ${timeoutMs}ms`);
                logToServer('error', `Model loading timed out after ${timeoutMs}ms`);
                loadingScreen.innerHTML = '<p>モデルの読み込みがタイムアウトしました。</p>';
                reject(new Error(`Model loading timed out after ${timeoutMs}ms`));
            }, timeoutMs);

            // Function to clear the timeout when loading completes
            const clearLoadingTimeout = () => {
                console.log('Clearing model loading timeout');
                logToServer('model_loading', 'Clearing model loading timeout');
                clearTimeout(timeoutId);
            };
            // Show loading progress with enhanced visualization
            const loadingScreen = document.createElement('div');
            loadingScreen.style.position = 'absolute';
            loadingScreen.style.top = '50%';
            loadingScreen.style.left = '50%';
            loadingScreen.style.transform = 'translate(-50%, -50%)';
            loadingScreen.style.textAlign = 'center';
            loadingScreen.style.color = '#666';
            loadingScreen.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
            loadingScreen.style.padding = '20px';
            loadingScreen.style.borderRadius = '10px';
            loadingScreen.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
            loadingScreen.style.zIndex = '1000';
            loadingScreen.style.width = '300px';

            // Create progress bar container
            const progressBarContainer = document.createElement('div');
            progressBarContainer.style.width = '100%';
            progressBarContainer.style.height = '20px';
            progressBarContainer.style.backgroundColor = '#f0f0f0';
            progressBarContainer.style.borderRadius = '10px';
            progressBarContainer.style.overflow = 'hidden';
            progressBarContainer.style.marginTop = '10px';
            progressBarContainer.style.marginBottom = '10px';
            progressBarContainer.style.border = '1px solid #ddd';

            // Create progress bar
            const progressBar = document.createElement('div');
            progressBar.id = 'vrm-progress-bar';
            progressBar.style.width = '0%';
            progressBar.style.height = '100%';
            progressBar.style.backgroundColor = '#4285f4';
            progressBar.style.transition = 'width 0.3s ease-in-out';
            progressBarContainer.appendChild(progressBar);

            // Create text display for percentage
            const progressText = document.createElement('div');
            progressText.id = 'loading-progress';
            progressText.textContent = '0%';
            progressText.style.fontWeight = 'bold';
            progressText.style.marginTop = '5px';

            // Add file info
            const fileInfo = document.createElement('div');
            fileInfo.id = 'file-info';
            fileInfo.style.fontSize = '12px';
            fileInfo.style.marginTop = '10px';
            fileInfo.style.color = '#888';

            // Add loading status
            const loadingStatus = document.createElement('div');
            loadingStatus.id = 'loading-status';
            loadingStatus.textContent = 'ファイルをダウンロード中...';
            loadingStatus.style.fontSize = '14px';
            loadingStatus.style.marginBottom = '10px';

            // Add time estimate
            const timeEstimate = document.createElement('div');
            timeEstimate.id = 'time-estimate';
            timeEstimate.style.fontSize = '12px';
            timeEstimate.style.marginTop = '10px';
            timeEstimate.style.color = '#888';

            // Assemble the loading screen
            loadingScreen.innerHTML = '<p style="font-size: 16px; margin-bottom: 10px;">VRMモデルを読み込み中...</p>';
            loadingScreen.appendChild(loadingStatus);
            loadingScreen.appendChild(progressBarContainer);
            loadingScreen.appendChild(progressText);
            loadingScreen.appendChild(fileInfo);
            loadingScreen.appendChild(timeEstimate);

            // Add to DOM
            this.canvas.parentNode.appendChild(loadingScreen);

            // Track loading start time for time estimation
            const loadStartTime = Date.now();

            try {
                // Log available Babylon.js components for debugging
                console.log('Checking Babylon.js components availability:');
                logToServer('model_loading', 'Checking Babylon.js components availability');

                const babylonAvailable = typeof BABYLON !== 'undefined';
                const sceneLoaderAvailable = babylonAvailable && typeof BABYLON.SceneLoader !== 'undefined';
                const gltf2Available = babylonAvailable && typeof BABYLON.GLTF2 !== 'undefined';
                const gltfLoaderAvailable = gltf2Available && typeof BABYLON.GLTF2.GLTFLoader !== 'undefined';

                console.log('- BABYLON available:', babylonAvailable);
                console.log('- BABYLON.SceneLoader available:', sceneLoaderAvailable);
                console.log('- BABYLON.GLTF2 available:', gltf2Available);
                console.log('- BABYLON.GLTF2.GLTFLoader available:', gltfLoaderAvailable);

                logToServer('model_loading', 'Babylon.js components check', {
                    babylonAvailable,
                    sceneLoaderAvailable,
                    gltf2Available,
                    gltfLoaderAvailable
                });

                if (!babylonAvailable) {
                    console.error('Babylon.js not found. Make sure Babylon.js is properly loaded.');
                    logToServer('error', 'Babylon.js not found. Make sure Babylon.js is properly loaded.');
                    loadingScreen.innerHTML = '<p>Babylon.jsが見つかりません。</p>';
                    reject(new Error('Babylon.js not found'));
                    return;
                }

                if (!sceneLoaderAvailable) {
                    console.error('BABYLON.SceneLoader not found. Make sure Babylon.js loaders are properly loaded.');
                    logToServer('error', 'BABYLON.SceneLoader not found. Make sure Babylon.js loaders are properly loaded.');
                    loadingScreen.innerHTML = '<p>Babylon.jsローダーが見つかりません。</p>';
                    reject(new Error('BABYLON.SceneLoader not found'));
                    return;
                }

                // Check if babylon-vrm-loader is available
                if (!gltf2Available || !gltfLoaderAvailable) {
                    console.error('BABYLON.GLTF2.GLTFLoader not found. Make sure babylon-vrm-loader is properly loaded.');
                    logToServer('error', 'BABYLON.GLTF2.GLTFLoader not found. Make sure babylon-vrm-loader is properly loaded.');
                    loadingScreen.innerHTML = '<p>VRMローダーが見つかりません。</p>';
                    reject(new Error('BABYLON.GLTF2.GLTFLoader not found'));
                    return;
                }

                // Check if VRM loader is registered with SceneLoader
                console.log('Checking if VRM loader is registered with SceneLoader');
                logToServer('model_loading', 'Checking if VRM loader is registered with SceneLoader');

                let vrmLoaderRegistered = false;
                let registeredPluginNames = [];

                if (BABYLON.SceneLoader.RegisteredPlugins) {
                    console.log('Registered plugins:', BABYLON.SceneLoader.RegisteredPlugins);
                    registeredPluginNames = BABYLON.SceneLoader.RegisteredPlugins.map(plugin => plugin.name);
                    logToServer('model_loading', 'Registered plugins', { plugins: registeredPluginNames });

                    vrmLoaderRegistered = BABYLON.SceneLoader.RegisteredPlugins.some(plugin => 
                        plugin.name === 'gltf' || plugin.name === 'vrm');
                } else {
                    console.warn('No plugins registered with SceneLoader');
                    logToServer('warning', 'No plugins registered with SceneLoader');
                }

                console.log('VRM loader registered:', vrmLoaderRegistered);
                logToServer('model_loading', `VRM loader registered: ${vrmLoaderRegistered}`);

                if (!vrmLoaderRegistered) {
                    console.warn('VRM loader may not be properly registered. Attempting to continue anyway.');
                    logToServer('warning', 'VRM loader may not be properly registered. Attempting to continue anyway.');
                    loadingScreen.innerHTML = '<p>VRMローダーが正しく登録されていない可能性があります。続行を試みます...</p>';
                } else {
                    console.log('VRM loader is properly registered');
                    logToServer('model_loading', 'VRM loader is properly registered');
                }

                // Create a ground for the character to stand on
                const ground = BABYLON.MeshBuilder.CreateGround("ground", {width: 6, height: 6}, this.scene);
                ground.position.y = -0.01; // Slightly below the character
                ground.visibility = 0.3; // Semi-transparent

                // Set up loading progress callback
                BABYLON.SceneLoader.OnPluginActivatedObservable.add(function (loader) {
                    if (loader.name === "gltf") {
                        loader.onParsed = function (data) {
                            console.log("Parsing complete: all textures and materials are created");
                            loadingScreen.innerHTML = '<p>VRMモデルを処理中...</p><p>テクスチャとマテリアルを準備しています</p>';
                        };
                        loader.onProgress = function (event) {
                            if (event.lengthComputable) {
                                const progress = Math.round((event.loaded * 100) / event.total);
                                document.getElementById('loading-progress').textContent = `${progress}%`;
                            }
                        };
                    }
                });

                // Extract the directory and filename from the modelPath
                let modelDir = '';
                let modelFilename = modelPath;

                // Handle both absolute and relative paths
                if (modelPath.includes('/')) {
                    const lastSlashIndex = modelPath.lastIndexOf('/');
                    modelDir = modelPath.substring(0, lastSlashIndex + 1);
                    modelFilename = modelPath.substring(lastSlashIndex + 1);
                }

                // Check if .vrm extension is registered with SceneLoader
                console.log('Checking if .vrm extension is registered with SceneLoader');
                logToServer('model_loading', 'Checking if .vrm extension is registered with SceneLoader');

                const registeredExtensions = [];
                if (BABYLON.SceneLoader.RegisteredPlugins) {
                    for (const plugin of BABYLON.SceneLoader.RegisteredPlugins) {
                        if (plugin.extensions) {
                            registeredExtensions.push(...plugin.extensions);
                        }
                    }
                }

                console.log('Registered file extensions:', registeredExtensions);
                logToServer('model_loading', 'Registered file extensions', { extensions: registeredExtensions });

                const vrmExtensionRegistered = registeredExtensions.includes('.vrm');
                console.log('VRM extension registered:', vrmExtensionRegistered);
                logToServer('model_loading', `VRM extension registered: ${vrmExtensionRegistered}`);

                if (!vrmExtensionRegistered) {
                    console.warn('VRM file extension is not registered with SceneLoader. Attempting to register it.');
                    logToServer('warning', 'VRM file extension is not registered with SceneLoader. Attempting to register it.');

                    // Try to register VRM extension with the GLTF loader if available
                    if (BABYLON.GLTF2 && BABYLON.GLTF2.GLTFLoader) {
                        try {
                            console.log('Attempting to manually register .vrm extension with GLTF loader');
                            logToServer('model_loading', 'Attempting to manually register .vrm extension with GLTF loader');

                            // This is a workaround - we're trying to register .vrm extension with the GLTF loader
                            BABYLON.SceneLoader.RegisterPlugin({
                                name: "gltf",
                                extensions: [".vrm"],
                                canDirectLoad: function(data) {
                                    return data.indexOf("asset") !== -1;
                                },
                                importMesh: function(meshesNames, scene, data, rootUrl, meshes, particleSystems, skeletons, onSuccess) {
                                    var loaderOptions = {
                                        skipMaterials: false
                                    };
                                    var loader = new BABYLON.GLTF2.GLTFLoader(scene);
                                    loader.importMeshAsync(meshesNames, rootUrl, data, null, loaderOptions).then(function(result) {
                                        if (onSuccess) {
                                            onSuccess(result.meshes, particleSystems, skeletons);
                                        }
                                    });
                                    return true;
                                },
                                load: function(scene, data, rootUrl, onSuccess, onProgress, onError) {
                                    var loaderOptions = {
                                        skipMaterials: false
                                    };
                                    var loader = new BABYLON.GLTF2.GLTFLoader(scene);
                                    loader.importMeshAsync(null, rootUrl, data, null, loaderOptions).then(function() {
                                        if (onSuccess) {
                                            onSuccess();
                                        }
                                    });
                                    return true;
                                }
                            });
                            console.log('Manually registered .vrm extension with GLTF loader');
                            logToServer('model_loading', 'Successfully registered .vrm extension with GLTF loader');

                            // Verify registration was successful
                            const updatedExtensions = [];
                            if (BABYLON.SceneLoader.RegisteredPlugins) {
                                for (const plugin of BABYLON.SceneLoader.RegisteredPlugins) {
                                    if (plugin.extensions) {
                                        updatedExtensions.push(...plugin.extensions);
                                    }
                                }
                            }
                            const registrationSuccessful = updatedExtensions.includes('.vrm');
                            console.log('Registration verification - VRM extension now registered:', registrationSuccessful);
                            logToServer('model_loading', `Registration verification - VRM extension now registered: ${registrationSuccessful}`);

                        } catch (regError) {
                            console.error('Failed to register VRM extension:', regError);
                            logToServer('error', 'Failed to register VRM extension', {
                                error: regError.message,
                                stack: regError.stack
                            });
                        }
                    } else {
                        console.error('Cannot register VRM extension: BABYLON.GLTF2.GLTFLoader not available');
                        logToServer('error', 'Cannot register VRM extension: BABYLON.GLTF2.GLTFLoader not available');
                    }
                } else {
                    console.log('VRM extension is already registered with SceneLoader');
                    logToServer('model_loading', 'VRM extension is already registered with SceneLoader');
                }

                // Log the model path for debugging
                console.log('Loading VRM model from:', modelDir + modelFilename);

                // Update file info without destroying the entire loading screen
                const fileInfo = document.getElementById('file-info');
                if (fileInfo) {
                    fileInfo.textContent = `ファイル: ${modelFilename}`;
                }

                // Reset progress indicators
                const progressBar = document.getElementById('vrm-progress-bar');
                if (progressBar) progressBar.style.width = '0%';

                const progressText = document.getElementById('loading-progress');
                if (progressText) progressText.textContent = '0%';

                const loadingStatus = document.getElementById('loading-status');
                if (loadingStatus) {
                    loadingStatus.textContent = 'ファイルをダウンロード中...';
                }

                // Log the start of model loading
                console.log('Starting to load VRM model using BABYLON.SceneLoader.Append');
                logToServer('model_loading', 'Starting to load VRM model using BABYLON.SceneLoader.Append', {
                    modelDir,
                    modelFilename,
                    fullPath: modelDir + modelFilename
                });

                // Use the babylon-vrm-loader to load the VRM model
                BABYLON.SceneLoader.Append(modelDir, modelFilename, this.scene, (scene) => {
                    // Clear the loading timeout
                    clearLoadingTimeout();

                    console.log('VRM model loaded successfully, parsing complete');
                    logToServer('model_loading', 'VRM model loaded successfully, parsing complete');

                    // Update loading status before removing
                    const loadingStatus = document.getElementById('loading-status');
                    if (loadingStatus) {
                        loadingStatus.textContent = 'モデルの初期化完了！';
                    }

                    // Update progress to 100% for completion
                    const progressBar = document.getElementById('vrm-progress-bar');
                    if (progressBar) progressBar.style.width = '100%';

                    const progressText = document.getElementById('loading-progress');
                    if (progressText) progressText.textContent = '100%';

                    // Show completion message briefly before removing
                    setTimeout(() => {
                        loadingScreen.remove();
                    }, 500);

                    // Log detailed scene information
                    const sceneInfo = {
                        meshes: scene.meshes.length,
                        materials: scene.materials.length,
                        textures: scene.textures.length,
                        skeletons: scene.skeletons.length,
                        animationGroups: scene.animationGroups.length
                    };

                    console.log('Model loaded, scene contains:', sceneInfo);
                    console.log('- Meshes:', scene.meshes.length);
                    console.log('- Materials:', scene.materials.length);
                    console.log('- Textures:', scene.textures.length);
                    console.log('- Skeletons:', scene.skeletons.length);
                    console.log('- Animation Groups:', scene.animationGroups.length);

                    logToServer('model_loading', 'Model loaded successfully, scene details', sceneInfo);

                    // Find the root mesh of the loaded model
                    const rootMeshes = scene.meshes.filter(mesh => !mesh.parent && mesh.name !== "ground");
                    console.log('Root meshes found:', rootMeshes.length);

                    if (rootMeshes.length === 0) {
                        console.error('No root meshes were found in the loaded model');
                        reject(new Error('No meshes were loaded'));
                        return;
                    }

                    // Store the VRM model
                    this.vrm = {
                        meshes: scene.meshes.filter(mesh => mesh.name !== "ground"),
                        skeletons: scene.skeletons,
                        animationGroups: scene.animationGroups,
                        rootMesh: rootMeshes[0]
                    };

                    // Position the model
                    this.vrm.rootMesh.position = new BABYLON.Vector3(0, 0, 0);

                    // Look for morph targets (blend shapes) for facial animations
                    this.findMorphTargets();

                    console.log("VRM model loaded successfully:", this.vrm);
                    resolve(this.vrm);
                }, (progress) => {
                    // Progress callback with enhanced visualization
                    if (progress.lengthComputable) {
                        const progressPercent = Math.round((progress.loaded * 100) / progress.total);
                        console.log(`Loading model: ${progressPercent}%`);

                        // Log progress at key milestones (25%, 50%, 75%, 100%)
                        if (progressPercent % 25 === 0 || progressPercent === 100) {
                            const loadedMB = (progress.loaded / (1024 * 1024)).toFixed(2);
                            const totalMB = (progress.total / (1024 * 1024)).toFixed(2);
                            logToServer('model_loading', `Model download progress: ${progressPercent}%`, {
                                loaded: loadedMB + 'MB',
                                total: totalMB + 'MB',
                                filename: modelFilename
                            });
                        }

                        // Update text percentage
                        const progressText = document.getElementById('loading-progress');
                        if (progressText) progressText.textContent = `${progressPercent}%`;

                        // Update progress bar
                        const progressBar = document.getElementById('vrm-progress-bar');
                        if (progressBar) progressBar.style.width = `${progressPercent}%`;

                        // Update file info with size information
                        const fileInfo = document.getElementById('file-info');
                        if (fileInfo) {
                            const loadedMB = (progress.loaded / (1024 * 1024)).toFixed(2);
                            const totalMB = (progress.total / (1024 * 1024)).toFixed(2);
                            fileInfo.textContent = `${loadedMB}MB / ${totalMB}MB (${modelFilename})`;
                        }

                        // Update loading status based on progress
                        const loadingStatus = document.getElementById('loading-status');
                        let statusText = '';
                        if (loadingStatus) {
                            if (progressPercent < 25) {
                                statusText = 'ファイルをダウンロード中...';
                                loadingStatus.textContent = statusText;
                            } else if (progressPercent < 75) {
                                statusText = 'モデルデータを処理中...';
                                loadingStatus.textContent = statusText;
                            } else {
                                statusText = 'テクスチャとマテリアルを準備中...';
                                loadingStatus.textContent = statusText;
                            }

                            // Log status changes
                            if (progressPercent === 25 || progressPercent === 75) {
                                logToServer('model_loading', `Loading status update: ${statusText} (${progressPercent}%)`);
                            }
                        }

                        // Update estimated time remaining
                        const timeEstimate = document.getElementById('time-estimate');
                        if (timeEstimate && loadStartTime) {
                            const elapsedTime = (Date.now() - loadStartTime) / 1000;
                            const estimatedTotalTime = elapsedTime / (progressPercent / 100);
                            const remainingTime = Math.max(0, estimatedTotalTime - elapsedTime).toFixed(0);

                            if (remainingTime > 0) {
                                timeEstimate.textContent = `推定残り時間: 約${remainingTime}秒`;
                            }
                        }
                    } else {
                        console.log('Loading progress not computable');
                        logToServer('warning', 'Model loading progress not computable');
                    }
                }, (scene, message, exception) => {
                    // Clear the loading timeout
                    clearLoadingTimeout();

                    console.error(`Error loading VRM model: ${message}`);

                    // Error callback - update loading screen with error before removing
                    const loadingStatus = document.getElementById('loading-status');
                    if (loadingStatus) {
                        loadingStatus.textContent = 'エラーが発生しました';
                        loadingStatus.style.color = '#ea4335';
                    }

                    // Collect detailed error information
                    const errorDetails = {
                        message: message,
                        modelPath: modelPath,
                        modelDir: modelDir,
                        modelFilename: modelFilename,
                        browserInfo: navigator.userAgent,
                        webglInfo: this.getWebGLInfo(),
                        time: new Date().toISOString()
                    };

                    // Add Babylon.js version information if available
                    if (BABYLON && BABYLON.Engine && BABYLON.Engine.Version) {
                        errorDetails.babylonVersion = BABYLON.Engine.Version;
                    }

                    // Add VRM loader information if available
                    if (BABYLON && BABYLON.GLTF2) {
                        errorDetails.gltf2Available = true;
                        if (BABYLON.GLTF2.GLTFLoader) {
                            errorDetails.gltfLoaderAvailable = true;
                        }
                    }

                    // Add registered plugins information
                    if (BABYLON && BABYLON.SceneLoader && BABYLON.SceneLoader.RegisteredPlugins) {
                        errorDetails.registeredPlugins = BABYLON.SceneLoader.RegisteredPlugins.map(plugin => plugin.name);
                    }

                    if (exception) {
                        errorDetails.exception = exception.toString();
                        if (exception.stack) {
                            errorDetails.stack = exception.stack;
                        }
                        console.error('Exception details:', exception);
                    }

                    // Log the error to the server with detailed information
                    logToServer('error', `Error loading VRM model: ${message}`, errorDetails);

                    // Add more user-friendly error message based on the error type
                    let userErrorMessage = 'VRMモデルの読み込み中にエラーが発生しました。';

                    if (message.includes('NetworkError') || message.includes('Failed to fetch') || message.includes('CORS')) {
                        userErrorMessage += 'ネットワークエラーが発生しました。インターネット接続を確認してください。';
                        console.error('Network error detected while loading the model');
                        logToServer('error', 'Network error detected while loading the model');
                    } else if (message.includes('memory') || message.includes('out of memory')) {
                        userErrorMessage += 'メモリ不足エラーが発生しました。ブラウザを再起動するか、他のタブを閉じてみてください。';
                        console.error('Memory error detected while loading the model');
                        logToServer('error', 'Memory error detected while loading the model');
                    } else if (message.includes('WebGL') || message.includes('GPU')) {
                        userErrorMessage += 'グラフィックスエラーが発生しました。ブラウザのWebGL設定を確認してください。';
                        console.error('WebGL/GPU error detected while loading the model');
                        logToServer('error', 'WebGL/GPU error detected while loading the model');
                    } else if (message.includes('parse') || message.includes('syntax') || message.includes('invalid')) {
                        userErrorMessage += 'モデルファイルの解析エラーが発生しました。ファイルが破損している可能性があります。';
                        console.error('Parse error detected while loading the model');
                        logToServer('error', 'Parse error detected while loading the model');
                    }

                    // Update the loading screen with the user-friendly error message
                    const errorMessageElement = document.createElement('p');
                    errorMessageElement.textContent = userErrorMessage;
                    errorMessageElement.style.color = '#ea4335';
                    errorMessageElement.style.marginTop = '10px';
                    loadingScreen.appendChild(errorMessageElement);

                    // Show error briefly before removing
                    setTimeout(() => {
                        loadingScreen.remove();
                    }, 3000); // Show for 3 seconds to give user time to read

                    // Error already logged above, no need to duplicate
                    if (exception) {
                        console.error('Exception details:', exception);
                        if (exception.stack) {
                            console.error('Stack trace:', exception.stack);
                        }
                    }
                    reject(new Error(`Failed to load VRM model: ${message}`));
                });
            } catch (error) {
                // Clear the loading timeout
                clearLoadingTimeout();

                // Remove loading screen
                loadingScreen.remove();
                console.error('Error in loadVRMModel:', error);
                reject(error);
            }
        });
    }

    // Find morph targets (blend shapes) for facial animations
    findMorphTargets() {
        if (!this.vrm || !this.vrm.meshes) return;

        this.morphTargets = {
            blink: null,
            mouth: null
        };

        // Look through all meshes for morph targets
        for (const mesh of this.vrm.meshes) {
            if (mesh.morphTargetManager) {
                // Look for blink and mouth morph targets
                for (let i = 0; i < mesh.morphTargetManager.numTargets; i++) {
                    const target = mesh.morphTargetManager.getTarget(i);
                    const name = target.name.toLowerCase();

                    if (name.includes('blink') || name.includes('eye') || name.includes('まばたき')) {
                        this.morphTargets.blink = {
                            mesh: mesh,
                            index: i
                        };
                        console.log(`Found blink morph target: ${target.name}`);
                    }

                    if (name.includes('a') || name.includes('mouth') || name.includes('口') || name.includes('あ')) {
                        this.morphTargets.mouth = {
                            mesh: mesh,
                            index: i
                        };
                        console.log(`Found mouth morph target: ${target.name}`);
                    }
                }
            }
        }
    }

    // Start the render loop
    startRenderLoop() {
        console.log('Starting render loop');
        logToServer('init_step', 'Starting render loop');

        try {
            this.engine.runRenderLoop(() => {
                // Update facial animations
                this.updateFacialAnimations();

                // Render the scene
                this.scene.render();
            });

            console.log('Render loop started successfully');
            logToServer('init_step', 'Render loop started successfully');

            // Handle window resize
            window.addEventListener('resize', () => {
                this.engine.resize();
            });

            console.log('Window resize handler registered');
            logToServer('init_step', 'Window resize handler registered');
        } catch (error) {
            console.error('Error starting render loop:', error);
            logToServer('error', 'Error starting render loop', {
                error: error.message,
                stack: error.stack
            });
        }
    }

    // Update facial animations (blinking and mouth movement)
    updateFacialAnimations() {
        if (!this.vrm || !this.morphTargets) return;

        // Update blinking
        if (this.morphTargets.blink && this.morphTargets.blink.mesh) {
            const blinkValue = this.isBlinking ? 1.0 : 0.0;
            this.morphTargets.blink.mesh.morphTargetManager.getTarget(this.morphTargets.blink.index).influence = blinkValue;
        }

        // Update mouth movement for speaking
        if (this.morphTargets.mouth && this.morphTargets.mouth.mesh) {
            if (this.animationState === 'speaking') {
                // Oscillate mouth open value for speaking effect
                this.mouthOpenValue = 0.3 + 0.2 * Math.sin(Date.now() * 0.01);
            } else {
                // Close mouth when not speaking
                this.mouthOpenValue = 0;
            }
            this.morphTargets.mouth.mesh.morphTargetManager.getTarget(this.morphTargets.mouth.index).influence = this.mouthOpenValue;
        }
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
        // Implement emotion reactions when morph targets are identified
    }

    // Show a fallback message when the model can't be loaded
    showFallbackMessage(message) {
        // Hide the canvas
        this.canvas.style.display = 'none';

        // Log the fallback message to the server
        logToServer('warning', `Showing fallback message to user: ${message}`);

        // Create a more informative fallback message with troubleshooting options
        const fallbackHtml = `
            <div style="padding: 20px; text-align: center; color: #666; background-color: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);">
                <h3 style="margin-bottom: 15px; color: #333;">キャラクター読み込みエラー</h3>
                <p style="margin-bottom: 10px;">${message}</p>
                <p style="margin-bottom: 10px;">static/models/pneuma/model.vrm を配置してください。</p>
                <div style="margin-top: 20px; padding: 15px; background-color: #f1f1f1; border-radius: 8px; text-align: left;">
                    <p style="font-weight: bold; margin-bottom: 10px;">トラブルシューティング:</p>
                    <ul style="list-style-type: disc; padding-left: 20px; margin-bottom: 10px;">
                        <li>ブラウザを更新してみてください</li>
                        <li>別のブラウザ（Chrome、Firefox、Edgeなど）を試してみてください</li>
                        <li>WebGLが有効になっていることを確認してください</li>
                        <li>VRMモデルファイルが正しく配置されていることを確認してください</li>
                    </ul>
                    <p style="margin-top: 15px;">
                        <button id="retry-character-load" style="padding: 8px 15px; background-color: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer;">
                            再試行
                        </button>
                        <button id="use-fallback-model" style="margin-left: 10px; padding: 8px 15px; background-color: #f1f1f1; color: #333; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;">
                            フォールバックモデルを使用
                        </button>
                    </p>
                </div>
            </div>
        `;

        // Show the message in the container
        this.canvas.parentNode.innerHTML = fallbackHtml;

        // Add event listeners to the buttons
        setTimeout(() => {
            const retryButton = document.getElementById('retry-character-load');
            const fallbackButton = document.getElementById('use-fallback-model');

            if (retryButton) {
                retryButton.addEventListener('click', () => {
                    logToServer('info', 'User clicked retry button');
                    window.location.reload();
                });
            }

            if (fallbackButton) {
                fallbackButton.addEventListener('click', () => {
                    logToServer('info', 'User clicked fallback model button');
                    // Add fallback model parameter to URL and reload
                    const url = new URL(window.location.href);
                    url.searchParams.set('forceFallback', 'true');
                    window.location.href = url.toString();
                });
            }
        }, 100);
    }

    // Handle window resize
    resize() {
        if (!this.isInitialized || !this.engine) return;
        this.engine.resize();
    }

    // Get WebGL information for debugging
    getWebGLInfo() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

            if (!gl) {
                return { supported: false, message: 'WebGL not supported' };
            }

            const info = {
                supported: true,
                vendor: gl.getParameter(gl.VENDOR),
                renderer: gl.getParameter(gl.RENDERER),
                version: gl.getParameter(gl.VERSION),
                shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
                maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
                extensions: gl.getSupportedExtensions()
            };

            return info;
        } catch (e) {
            return { 
                supported: false, 
                message: 'Error getting WebGL info', 
                error: e.message 
            };
        }
    }

    // Check if a model file exists on the server
    async checkModelFileExists(modelPath) {
        try {
            // Remove leading slash if present for the API call
            const path = modelPath.startsWith('/') ? modelPath.substring(1) : modelPath;

            // Call the API to check if the file exists
            const response = await fetch(`/api/check_model_file?path=${encodeURIComponent(path)}`);
            const data = await response.json();

            if (data.status === 'success') {
                if (data.exists) {
                    console.log(`Model file exists: ${modelPath}`, data.file_info);
                    logToServer('info', `Model file exists: ${modelPath}`, data.file_info);
                    return true;
                } else {
                    console.warn(`Model file does not exist: ${modelPath}`);
                    logToServer('warning', `Model file does not exist: ${modelPath}`);
                    return false;
                }
            } else {
                console.error(`Error checking model file: ${data.message}`);
                logToServer('error', `Error checking model file: ${data.message}`);
                // Return true to continue trying to load the model anyway
                return true;
            }
        } catch (error) {
            console.error('Error checking if model file exists:', error);
            logToServer('error', 'Error checking if model file exists', {error: error.message, stack: error.stack});
            // Return true to continue trying to load the model anyway
            return true;
        }
    }
}

// Global character instance
let character = null;

// Function to initialize the character
function initializeCharacter(options = {}) {
    // Check if the container and canvas elements exist
    const characterCanvas = document.getElementById('character-canvas');

    if (!characterCanvas) {
        console.error('Character canvas element not found in the DOM');
        logToServer('error', 'Character canvas element not found in the DOM');
        return false;
    }

    // Default options
    const defaultOptions = {
        debug: false,
        forceFallbackModel: false
    };

    // Merge default options with provided options
    const mergedOptions = {...defaultOptions, ...options};

    // Create and initialize the character with options
    character = new BabylonCharacter('character-canvas', mergedOptions);

    // Initialize the character and set up event listeners
    character.initialize().then(success => {
        if (success) {
            console.log('Babylon VRM character initialized successfully');
            logToServer('success', 'Babylon VRM character initialized successfully and event listeners set up');

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
                logToServer('info', `Character speaking animation started, duration: ${duration}ms`);
            });

            document.addEventListener('audio-play-end', () => {
                character.idle();
                logToServer('info', 'Character returned to idle animation');
            });
        } else {
            console.warn('Failed to initialize Babylon VRM character');
            logToServer('warning', 'Failed to initialize Babylon VRM character');
        }
    }).catch(error => {
        console.error('Error during character initialization:', error);
        logToServer('error', 'Error during character initialization', {error: error.message, stack: error.stack});
    });

    return true;
}

// Initialize the character when the DOM is loaded and all scripts are ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM content loaded, waiting for window load event');

    // Wait for the window load event to ensure all scripts are loaded
    window.addEventListener('load', () => {
        console.log('Window loaded, checking for Babylon.js and VRM loader');

        // Parse URL parameters to check for debug options
        const urlParams = new URLSearchParams(window.location.search);
        const debugMode = urlParams.get('debug') === 'true';
        const forceFallback = urlParams.get('forceFallback') === 'true';
        const delayMs = parseInt(urlParams.get('delay') || '2000', 10);

        // Debug options
        const options = {
            debug: debugMode,
            forceFallbackModel: forceFallback
        };

        // Log debug options if enabled
        if (debugMode) {
            console.log('Debug mode enabled with options:', options);
        }

        // Function to check if Babylon.js and VRM loader are ready
        const checkBabylonReady = () => {
            console.log('Checking if Babylon.js and VRM loader are ready...');

            // Check if BABYLON is available
            if (typeof BABYLON === 'undefined') {
                console.warn('Babylon.js not loaded yet, will try loading scripts dynamically');
                loadBabylonScripts(options);
                return false;
            }

            // Check if SceneLoader is available
            if (typeof BABYLON.SceneLoader === 'undefined') {
                console.warn('Babylon.js SceneLoader not loaded yet');
                return false;
            }

            // Check if GLTF2 loader is available
            if (typeof BABYLON.GLTF2 === 'undefined' || typeof BABYLON.GLTF2.GLTFLoader === 'undefined') {
                console.warn('Babylon.js GLTF2 loader not loaded yet');
                return false;
            }

            // Check if VRM loader is registered
            let vrmLoaderRegistered = false;
            if (BABYLON.SceneLoader.RegisteredPlugins) {
                const registeredExtensions = [];
                for (const plugin of BABYLON.SceneLoader.RegisteredPlugins) {
                    if (plugin.extensions) {
                        registeredExtensions.push(...plugin.extensions);
                    }
                }
                vrmLoaderRegistered = registeredExtensions.includes('.vrm') || 
                                     registeredExtensions.includes('.VRM');
            }

            if (!vrmLoaderRegistered) {
                console.warn('VRM loader not registered yet');

                // Try to manually register VRM extension
                try {
                    if (BABYLON.GLTF2 && BABYLON.GLTF2.GLTFLoader) {
                        BABYLON.SceneLoader.RegisterPlugin({
                            name: "gltf",
                            extensions: [".vrm"],
                            canDirectLoad: function(data) {
                                return data.indexOf("asset") !== -1;
                            },
                            importMesh: function(meshesNames, scene, data, rootUrl, meshes, particleSystems, skeletons, onSuccess) {
                                var loaderOptions = { skipMaterials: false };
                                var loader = new BABYLON.GLTF2.GLTFLoader(scene);
                                loader.importMeshAsync(meshesNames, rootUrl, data, null, loaderOptions).then(function(result) {
                                    if (onSuccess) {
                                        onSuccess(result.meshes, particleSystems, skeletons);
                                    }
                                });
                                return true;
                            },
                            load: function(scene, data, rootUrl, onSuccess, onProgress, onError) {
                                var loaderOptions = { skipMaterials: false };
                                var loader = new BABYLON.GLTF2.GLTFLoader(scene);
                                loader.importMeshAsync(null, rootUrl, data, null, loaderOptions).then(function() {
                                    if (onSuccess) {
                                        onSuccess();
                                    }
                                });
                                return true;
                            }
                        });
                        console.log('Manually registered .vrm extension with GLTF loader');
                        vrmLoaderRegistered = true;
                    }
                } catch (error) {
                    console.error('Failed to register VRM extension:', error);
                }
            }

            return vrmLoaderRegistered;
        };

        // Function to initialize with retry
        const initWithRetry = (retryCount = 0, maxRetries = 3) => {
            if (retryCount > maxRetries) {
                console.error(`Failed to initialize character after ${maxRetries} retries`);
                logToServer('error', `Failed to initialize character after ${maxRetries} retries`);

                // Show a fallback message in the character container
                const container = document.getElementById('character-container');
                if (container) {
                    container.innerHTML = `
                        <div style="padding: 20px; text-align: center; color: #666;">
                            <p>VRMモデルの読み込みに失敗しました。</p>
                            <p>ブラウザのコンソールでエラーを確認してください。</p>
                            <p>デバッグモードを有効にするには: <code>?debug=true&forceFallback=true</code> をURLに追加</p>
                        </div>
                    `;
                    logToServer('info', 'Displayed fallback message to user after max retries');
                }
                return;
            }

            if (checkBabylonReady()) {
                console.log(`Babylon.js and VRM loader are ready, initializing character (attempt ${retryCount + 1})`);
                logToServer('init', `Initializing character (attempt ${retryCount + 1})`);
                initializeCharacter(options);
            } else {
                console.log(`Babylon.js or VRM loader not ready yet, retrying in ${delayMs}ms (attempt ${retryCount + 1})`);
                logToServer('warning', `Babylon.js or VRM loader not ready yet, retrying in ${delayMs}ms (attempt ${retryCount + 1})`);
                setTimeout(() => initWithRetry(retryCount + 1, maxRetries), delayMs);
            }
        };

        // Start initialization with retry
        console.log(`Starting character initialization with ${delayMs}ms delay`);
        logToServer('init', `Starting character initialization with ${delayMs}ms delay`);
        setTimeout(() => initWithRetry(), delayMs);
    });
});

// Helper function to load Babylon.js scripts dynamically
function loadBabylonScripts(options = {}) {
    console.log('Loading Babylon.js scripts dynamically');

    // Load Babylon.js core
    const babylonScript = document.createElement('script');
    babylonScript.src = "https://cdn.babylonjs.com/babylon.js";
    babylonScript.onload = () => {
        console.log("Babylon.js core loaded dynamically");

        // Load Babylon.js loaders
        const loadersScript = document.createElement('script');
        loadersScript.src = "https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js";
        loadersScript.onload = () => {
            console.log("Babylon.js loaders loaded dynamically");

            // Load Babylon VRM Loader
            const vrmLoaderScript = document.createElement('script');
            vrmLoaderScript.src = "https://cdn.jsdelivr.net/npm/babylon-vrm-loader/dist/index.js";
            vrmLoaderScript.onload = () => {
                console.log("Babylon VRM Loader loaded dynamically");

                // Initialize character after all scripts are loaded with the provided options
                setTimeout(() => {
                    initializeCharacter(options);
                }, 500);
            };
            document.head.appendChild(vrmLoaderScript);
        };
        document.head.appendChild(loadersScript);
    };
    document.head.appendChild(babylonScript);
}

// Export for use in other scripts
window.GeminiCharacter = {
    BabylonCharacter,
    initializeCharacter,
    getInstance: () => character,
    // Add a method to initialize with debug options
    initializeWithOptions: (options) => initializeCharacter(options)
};

// Also make the character instance available directly on window for compatibility with existing code
window.character = character;
