import os

def thorough_optimization(directory):
    # Extension and versioning config
    extensions = ('.html', '.css', '.js')
    target_version = "v33.0"
    
    print(f"🚀 Starting thorough optimization in {directory}...")
    
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip system/large folders
        if any(skip in root for skip in ['.git', '.venv', 'chroma_db', '__pycache__']):
            continue
            
        for file in files:
            if file.lower().endswith(extensions):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # 1. Convert .png to .webp
                    new_content = content.replace('.png', '.webp')
                    
                    # 2. Update all version strings found in previous iterations to v33.0
                    # Common versions used: v=4, v=32.0, v=33.0
                    versions_to_replace = ['v=4', 'v=32.0', 'v=2.0', 'v=1.0', 'v=5']
                    for v in versions_to_replace:
                        new_content = new_content.replace(v, 'v=33.0')
                    
                    if new_content != original_content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"✅ Optimized: {os.path.relpath(path, directory)}")
                        count += 1
                except Exception as e:
                    print(f"❌ Error on {file}: {e}")
                    
    print(f"✨ Done! Optimized {count} files.")

if __name__ == "__main__":
    thorough_optimization('.')
