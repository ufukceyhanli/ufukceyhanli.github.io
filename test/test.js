function initializeViz() {
  // JS object that points at empty div in the html
  var placeholderDiv = document.getElementById("tableauViz");
  // URL of the viz to be embedded
  var url = "https://public.tableau.com/views/insights-to-crimes-in-czehia/Dashboard1?:language=en&:retry=yes&:display_count=y&:origin=viz_share_link";
  // An object that contains options specifying how to embed the viz
  var options = {
    width: '1000px',
    height: '4000px',
    hideTabs: true,
    hideToolbar: true,
  };
  viz = new tableau.Viz(placeholderDiv, url, options);
}