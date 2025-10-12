#!/bin/bash

HOOKS_DIR="hooks"
GIT_HOOKS_DIR=".git/hooks"

# Check if we're in a Git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a Git repository. Please run this script from the project root."
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

if [ -d "$HOOKS_DIR" ]; then
    echo "Installing Git hooks..."
    
    for hook in "$HOOKS_DIR"/*; do
        if [ -f "$hook" ]; then
            hook_name=$(basename "$hook")
            target="$GIT_HOOKS_DIR/$hook_name"
            
            # Create symlink
            if command -v ln >/dev/null 2>&1; then
                # Remove existing hook if it exists
                [ -e "$target" ] && rm "$target"
                
                # Create symlink using relative path
                ln -s "../../$hook" "$target"
                echo "Symlinked: $hook_name"
            else
                # Fallback to copying if ln is not available
                cp "$hook" "$target"
                echo "Copied: $hook_name"
            fi
            
            chmod +x "$target"
        fi
    done
    
    echo "Git hooks installation completed!"
    echo ""
    echo "Installed hooks:"
    ls -la "$GIT_HOOKS_DIR"
else
    echo "Error: $HOOKS_DIR directory not found."
    exit 1
fi