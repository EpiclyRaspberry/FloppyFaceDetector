from gui import FaceMeshApp

if __name__ == "__main__":
    try:
        app = FaceMeshApp()
        app.run()
    except Exception as e:
        print(f"Critical Error: {e}")