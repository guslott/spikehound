---
description: Add a new tab to the main application window
---

# Add a New GUI Tab

Follow these steps to add a new tab to the application's tab widget.

## Prerequisites
- Review `gui/gui_readme.md` for widget patterns
- Understand the tab's purpose and content

## Steps

1. **Create the tab widget file**
   ```
   gui/my_new_tab.py
   ```

2. **Implement the tab class**
   ```python
   """MyNewTab - Description of what the tab does."""
   from __future__ import annotations
   
   from typing import Optional
   from PySide6 import QtCore, QtWidgets
   
   
   class MyNewTab(QtWidgets.QWidget):
       """Tab for [feature description]."""
       
       # Define signals for external communication
       someAction = QtCore.Signal(object)
       
       def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
           super().__init__(parent)
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

3. **Register in MainWindow**
   
   In `gui/main_window.py`:
   
   a. Add import at top:
   ```python
   from .my_new_tab import MyNewTab
   ```
   
   b. Create and add tab in `_init_ui()`:
   ```python
   # Find where other tabs are added (look for addTab calls)
   self.my_tab = MyNewTab()
   self.tab_widget.addTab(self.my_tab, "My Tab")
   ```
   
   c. Connect signals if needed:
   ```python
   self.my_tab.someAction.connect(self._on_my_tab_action)
   ```
   
   d. Add handler method:
   ```python
   def _on_my_tab_action(self, data: object) -> None:
       """Handle action from my tab."""
       # Process the action
       pass
   ```

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
