quicklook = '''
{% extends base %}

<!-- goes in body -->
{% block postamble %}
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
{% endblock %}

<!-- goes in body -->
{% block contents %}
<div class="container">
  <div class="row">
    <div class="col col-sm-4">
      {{ embed(roots.brand) }}
    </div>
    <div class="col col-sm-4">
      {{ embed(roots.row1) }}
    </div>
    <div class="col col-sm-4">
      {{ embed(roots.row2) }}
    </div>
  </div>
  <hr/>
  <div class="row" style="height:50px">
    {{ embed(roots.info) }}
  </div>
  <div class="row">
    <div class="col col-sm-3">
        {{ embed(roots.metrics_selectors) }}
    </div>
    <div class="col col-sm-8 offset-sm-1">
        <div class="row d-flex justify-content-end">
            {{ embed(roots.view_switchers) }}
        </div>
        <div class="row">
            {{ embed(roots.plot_top) }}
        </div>
        <div class="row">
            {{ embed(roots.metrics_plots) }}
        </div>
    </div>
  </div>
</div>
{% endblock %}
'''
