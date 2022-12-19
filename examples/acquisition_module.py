from microEye.hardware import acquisition_module

try:
    import vimba as vb
except Exception:
    vb = None


if vb:
    with vb.Vimba.get_instance() as vimba:
        app, window = acquisition_module.StartGUI()

        app.exec_()
else:
    app, window = acquisition_module.StartGUI()

    app.exec_()
