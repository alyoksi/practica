python -m PyInstaller --clean ^
  --name Sandal ^
  --onedir ^
  --noconsole ^
  --collect-all numpy ^
  --collect-all scipy ^
  --collect-all skimage ^
  --collect-all numba ^
  --collect-all PySide6 ^
  --collect-all vtk ^
  --hidden-import vtkmodules.qt.QVTKRenderWindowInteractor ^
  --hidden-import vtkmodules.util.numpy_support ^
  gui_qt.py
