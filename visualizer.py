import os

try:
    from vtkmodules.vtkIOGeometry import vtkSTLReader
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
    from vtkmodules.vtkRenderingOpenGL2 import vtkRenderWindow
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch
    from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkCommonCore import vtkGlobalWarningDisplayOff, vtkOutputWindow
    VTK_MODERN = True
except ImportError:
    # fallback для старого API
    import vtk
    vtkSTLReader = vtk.vtkSTLReader
    vtkActor = vtk.vtkActor
    vtkPolyDataMapper = vtk.vtkPolyDataMapper
    vtkRenderer = vtk.vtkRenderer
    vtkRenderWindow = vtk.vtkRenderWindow
    vtkRenderWindowInteractor = vtk.vtkRenderWindowInteractor
    vtkPolyData = vtk.vtkPolyData
    vtkCubeSource = vtk.vtkCubeSource
    vtkOrientationMarkerWidget = vtk.vtkOrientationMarkerWidget
    vtkAxesActor = vtk.vtkAxesActor
    vtkTransform = vtk.vtkTransform
    vtkMatrix4x4 = vtk.vtkMatrix4x4
    vtkGlobalWarningDisplayOff = vtk.vtkObject.GlobalWarningDisplayOff
    vtkOutputWindow = vtk.vtkOutputWindow
    VTK_MODERN = False


class STLVisualizer:
    def __init__(self):
        # Suppress VTK output window
        vtkGlobalWarningDisplayOff()

        self.reader = vtkSTLReader()
        self.mapper = vtkPolyDataMapper()
        self.actor = vtkActor()
        self.renderer = vtkRenderer()
        self.render_window = None

        self.bbox_actor = None
        self.axes_widget = None
        self.bound_box_actor = None  # Rename to be consistent or change name

        self._setup_pipeline()

    def _setup_pipeline(self):
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0.0, 0.5, 1.0)  # Blue color for better contrast
        self.actor.GetProperty().SetAmbient(0.3)
        self.actor.GetProperty().SetDiffuse(0.7)
        self.actor.GetProperty().SetSpecular(0.5)
        self.actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.9, 0.9, 0.9)

    def set_render_window(self, render_window):
        self.render_window = render_window
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(400, 300)

    def load_stl(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        self.reader.SetFileName(filename)
        self.reader.Update()
        # Reset transform for new file
        self.actor.SetUserTransform(None)

        # Auto-adjust camera
        bounds = self.actor.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
        max_range = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(center)
        camera.SetPosition(
            center[0],
            center[1],
            center[2] + max_range * 1.5,
        )
        camera.SetViewUp(0, 1, 0)
        camera.SetClippingRange(0.01, max_range * 10)

        self.renderer.ResetCamera()
        self.render_window.Render()

    def get_render_window(self):
        return self.render_window

    def reset_view(self):
        self.renderer.ResetCamera()
        self.render_window.Render()

    def show_axes(self, show=True):
        if show:
            if self.axes_widget is None:
                axes = vtkAxesActor()
                self.axes_widget = vtkOrientationMarkerWidget()
                self.axes_widget.SetOrientationMarker(axes)
                self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
                self.axes_widget.InteractiveOff()
            if self.render_window_interactor:
                self.axes_widget.SetInteractor(self.render_window_interactor)
                self.axes_widget.EnabledOn()
        elif not show and self.axes_widget is not None:
            self.axes_widget.EnabledOff()

    def set_interactor(self, interactor):
        self.render_window_interactor = interactor
        if self.render_window_interactor:
            self.render_window_interactor.Initialize()
        self.render_window.Render()

    def reset_scene(self):
        """Reset the scene for loading a new model"""
        if self.bound_box_actor:
            self.renderer.RemoveActor(self.bound_box_actor)
            self.bound_box_actor = None
        if self.axes_widget:
            self.axes_widget.EnabledOff()
            self.axes_widget = None
        self.actor.SetUserTransform(None)

    def show_bounding_box(self, min_coords, max_coords):
        if self.bound_box_actor is not None:
            self.renderer.RemoveActor(self.bound_box_actor)

        import numpy as np

        min_coords = np.array(min_coords, dtype=float)
        max_coords = np.array(max_coords, dtype=float)
        center = (min_coords + max_coords) / 2
        sizes = max_coords - min_coords

        cube = vtkCubeSource()
        cube.SetXLength(sizes[0])
        cube.SetYLength(sizes[1])
        cube.SetZLength(sizes[2])

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(3)  # Make wireframe thicker
        actor.GetProperty().LightingOff()  # Disable lighting for consistent visibility

        actor.SetPosition(center)

        self.bound_box_actor = actor
        self.renderer.AddActor(actor)
        self.render_window.Render()

    def set_transform(self, rotation_matrix, origin=None):
        """Apply rotation matrix and translation to the actor"""
        if rotation_matrix is None:
            self.actor.SetUserTransform(None)
            self.renderer.ResetCamera()
            self.render_window.Render()
            return

        # Assuming rotation_matrix is 3x3 numpy array
        import numpy as np

        # Convert 3x3 to 4x4 homogeneous
        matrix_4x4 = np.eye(4)
        rotation_matrix_T = rotation_matrix.T
        matrix_4x4[:3, :3] = rotation_matrix_T

        # Add translation to center the model: translate by -origin, then rotate by rotation_matrix_T
        if origin is not None:
            matrix_4x4[:3, 3] = -np.dot(rotation_matrix_T, origin)

        # Create VTK matrix
        if VTK_MODERN:
            from vtkmodules.vtkCommonMath import vtkMatrix4x4
        else:
            vtkMatrix4x4 = vtk.vtkMatrix4x4

        vtk_matrix = vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, matrix_4x4[i, j])

        transform = vtkTransform()
        transform.SetMatrix(vtk_matrix)
        self.actor.SetUserTransform(transform)
        self.renderer.ResetCamera()
        self.render_window.Render()


if __name__ == "__main__":
    # Простой тест
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if os.path.exists(filename):
            print(f"Loading {filename}")
            visualizer = STLVisualizer()
            visualizer.load_stl(filename)
            # Создаём интерактор для standalone теста
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch
            iren = vtkRenderWindowInteractor()
            iren.SetRenderWindow(visualizer.render_window)
            style = vtkInteractorStyleSwitch()
            iren.SetInteractorStyle(style)
            iren.Initialize()
            iren.Start()
        else:
            print(f"File {filename} not found")
    else:
        print("Usage: python visualizer.py <stl_file>")
