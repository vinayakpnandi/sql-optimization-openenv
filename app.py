"""
Backward compatibility wrapper - imports from server.app for multi-mode deployment.
"""

from server.app import app, main

if __name__ == "__main__":
    main()
