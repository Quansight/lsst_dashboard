quicklook = """
{% extends base %}

<!-- goes in body -->
{% block postamble %}
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>

<style>

body {
  position: absolute;
  bottom: 0px;
  top: 0px;
  left: 0px;
  right: 0px;
}

.logo {
   width: 150px;
   height: auto;
}

.metricsRow {
    position:relative;
    top: 20px;
    bottom: 100px;
    left: 0;
    right: 0;
    overflow-y: auto;
}

.container-fluid {
  position: absolute;
  top:0;
  bottom:0;
  left:0;
  right:0;
}

.main-body-row {
  height:100%;
}


</style>

{% endblock %}

<!-- goes in body -->
{% block contents %}
<div class="container-fluid">

<nav class="navbar navbar-light bg-light">
  <span class="navbar-brand mb-0 h1">
    <img class="logo"
         src="https://www.lsst.org/sites/default/files/logos/LSST_web_white.png" />
      <i>Data Processing Explorer</i>
    </span>

    <ul class="nav navbar-nav navbar-left">
        <li>{{ embed(roots.row1) }}</li>
    </ul>
    <ul class="nav navbar-nav navbar-left">
        <li>{{ embed(roots.row2) }}</li>
    </ul>

</nav>

<div class="row">
  <div class="col col-sm-6">
  {{ embed(roots.flags) }}
    </div>
  <div class="col col-sm-6">
  {{ embed(roots.query_filter) }}
    </div>
</div>

<div class="row">
   <div class="col col-sm-12">
       {{ embed(roots.info) }}
   </div>
</div>
<hr />

  <div class="row main-body-row">
    <div class="col col-sm-4">
        {{ embed(roots.metrics_selectors) }}
    </div>

    <div class="col col-sm-offset-1 col-sm-7">

        <div class="row d-flex justify-content-end">
            {{ embed(roots.view_switchers) }}
        </div>

        {% if roots.plot_top %}
            <div class="row">
                {{ embed(roots.plot_top) }}
            </div>
            <div class="row metricsRow">
                {{ embed(roots.metrics_plots) }}
            </div>
        {% else %}
            <div class="row">
                {{ embed(roots.metrics_plots) }}
            </div>
        {% endif %}


        {% if 'priority' in data %}
            <p>Priority: {{ data['priority'] }}</p>
        {% endif %}

    </div>
  </div>
</div>
{% endblock %}
{% endblock %}
"""
