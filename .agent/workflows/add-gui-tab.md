---
description: Add a new tab to the main application window
---

# Add a New GUI Tab

Follow these steps to add a new plugin tab to the workspace dock.

## Prerequisites
- Review `gui/gui_readme.md` for widget patterns
- Understand the tab's purpose and content

## Steps

1. **Create the tab widget file**
   ```
   gui/tabs/my_new_tab.py
   ```

2. **Implement the tab class**
   ```python
   """MyNewTab - Description of what the tab does."""
   from __future__ import annotations
   
   from PySide6 import QtCore, QtWidgets
   from gui.tab_plugin_manager import BaseTab
   
   
   class MyNewTab(BaseTab):
       """Tab for [feature description]."""
       
       TAB_TITLE = "My Tab"

       # Define signals for external communication
       someAction = QtCore.Signal(object)
       
       def __init__(self, runtime, parent: QtWidgets.QWidget | None = None) -> None:
           super().__init__(runtime, parent)
           self._setup_ui()
           self._connect_signals()
       
       def _setup_ui(self) -> None:
           """Build the tab layout."""
           layout = QtWidgets.QVBoxLayout(self)
           
           # Add your widgets
           self.label = QtWidgets.QLabel("My Tab Content")
           layout.addWidget(self.label)
           
           self.button = QtWidgets.QPushButton("Do Something")
           layout.addWidget(self.button)
           
           layout.addStretch()  # Push content to top
       
       def _connect_signals(self) -> None:
           """Wire internal signals to handlers."""
           self.button.clicked.connect(self._on_button_clicked)
       
       def _on_button_clicked(self) -> None:
           """Handle button click."""
           self.someAction.emit("data")
       
       def refresh(self) -> None:
           """Called when tab becomes visible (optional)."""
           pass
   ```

3. **Load the plugin tab**
   
   Tabs in `gui/tabs/` that subclass `BaseTab` are auto-discovered by `TabPluginManager`.
   Restart the app and your tab will be added to `AnalysisDock` using `TAB_TITLE`.

4. **Test the tab**
   - Run the application
   - Verify tab appears in tab bar
   - Verify tab content displays correctly
   - Test any interactive elements

## Verification Checklist
- [ ] Tab appears in application
- [ ] Tab content renders correctly
- [ ] Interactive elements work
- [ ] Signals emit and are handled
- [ ] No console errors on tab switch
