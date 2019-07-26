import param
import panel as pn

class Application(param.Parameterized):
    """
    The Application class is the top-level object which coordinates
    the layout of the different components of the dashboard. It
    primarily consists of a header and body, the header is generated
    automatically from the current title and a logo while the body
    is a :class:`Component` which implements a :meth:`Component.panel`
    method which returns panel objects which can be composed into the
    overall application layout.
    """

    body = param.ClassSelector(class_=param.Parameterized, doc="""
        The Component to render in the application's body.""")

    logo = param.String('https://www.lsst.org/sites/default/files/logos/LSST_web_white.png', doc="""
        The logo to display in the header.""")

    title = param.String(doc="The current title to display in the header")

    def __init__(self, **params):
        super().__init__(**params)

        styles = {
            'white-space': 'nowrap',
        }

        text = '<h3><i>%s</i></h3>' % self.title

        self._title = pn.pane.HTML(text, margin=(0, 0),
                                   height=50, sizing_mode='stretch_width',
                                   style=styles)

        logo = pn.pane.PNG(self.logo, width=400 // 3, height=150 // 3)

        self.header = pn.Row(logo, self._title, max_height=50,)

    @param.depends('title', watch=True)
    def _update_title(self):
        "Upddates the title "
        self._title.object = '<h3><i>%s</i></h3>' % self.title

    @param.depends('body')
    def get_body(self):
        self.body.application = self
        return self.body.panel()
        # return self.body.panel_html_template()

    def render(self):
        "Renders the application as a single panel layout."
        self.body.application = self
        return pn.Column(self.get_body(),
                         width_policy='max', height_policy='max')
        # return self.get_body()


class Component(param.Parameterized):
    """
    Baseclass for visual components of the :class:`Application`.
    Defines a common API shared by all displayable components which
    allows access to the parent object, the :class:`Application` and
    defines methods which should return the title and a displayable
    panel object.
    """

    application = param.ClassSelector(class_=Application, precedence=-1, doc="""
        The :class:`Application` responsible for rendering the component.""")

    parent = param.ClassSelector(class_=param.Parameterized, precedence=-1, doc="""
        The parent :class:`Component` of the the object.""")

    label = param.String(default='Component', precedence=-1, doc="""
        A label to identify the Component by.""")

    __abstract = True

    def title(self):
        """
        Title to be displayed as part of the Application header.
        """
        return type(self).__name__

    def panel(self):
        """
        Should return a panel object to be rendered as part of the
        application.
        """


class TabComponent(Component):
    """
    The TabComponent displays a set of Tabs which may each contain a
    different :class:`Component`. This allows moving back and forth
    between stages of the :class:`Application`.
    """

    current = param.ClassSelector(class_=Component, precedence=-1, doc="""
        The component to display in the current tab..""")

    objects = param.List(doc="The list of sub-components to render in the tabs.")

    def __init__(self, *objects, **params):
        super().__init__(objects=list(objects), **params)
        self._layout = pn.Tabs(sizing_mode='stretch_both')
        self._layout.param.watch(self._update_title, 'active')

    def _update_title(self, event):
        "Updates the Application title depending on the selected tab."
        self.application.title = '<h1>%s</h1>' % self.objects[event.new].title()

    @param.depends('application', watch=True)
    def _init_panel(self):
        "Initializes the layout when the Application is initialized."
        for obj in self.objects:
            obj.parent = self
            obj.application = self.application
        self._layout[:] = [(obj.label, obj.panel()) for obj in self.objects]
        if self.current is None and self.objects:
            self.current = self.objects[0]

    @param.depends('current', watch=True)
    def _update_view(self):
        "Updates the current view whenever the current parameter is set."
        self.current.application = self.application
        obj_types = [type(obj) for obj in self.objects]
        title = self.current.title()
        if type(self.current) in obj_types:
            idx = obj_types.index(type(self.current))
            self.objects[idx] = self.current
            self._layout[idx] = (self.current.label, self.current.panel())
        else:
            self.objects.append(self.current)
            self._layout.append((self.current.label, self.current.panel()))
        self._layout.active = self.objects.index(self.current)
        self.application.title = title

    def panel(self):
        self._init_panel()
        return self._layout
