# VRM Character Model for Pneuma

This directory is where you should place your VRM character model file. The application is configured to look for a file named `model.vrm` in this directory.

## Adding Your VRM Model

1. Obtain a VRM model file. You can:
   - Create one using VRM-compatible software like VRoid Studio
   - Download free VRM models from sites like VRoid Hub or BOOTH
   - Commission a custom VRM model from an artist

2. Rename your VRM file to `model.vrm`

3. Place the file in this directory (`static/models/pneuma/model.vrm`)

4. Restart the application to see your character

## VRM Model Requirements

For best results, your VRM model should:

- Be optimized for web (file size under 10MB if possible)
- Include standard VRM blend shapes (especially "a" for mouth movement and "blink" for blinking)
- Have a T-pose or A-pose as the default pose
- Be properly rigged with a humanoid skeleton

## Troubleshooting

If your character doesn't appear:

1. Check the browser console for error messages
2. Verify that your VRM file is correctly named `model.vrm`
3. Make sure your VRM file is valid and not corrupted
4. Try a different VRM file to see if the issue is with your specific model

## Legal Considerations

When using VRM models:

1. Respect the license terms of the model you're using
2. If using a model you didn't create, ensure you have permission to use it
3. Some VRM models have usage restrictions (commercial use, modification, etc.)
