(() => {
  // <stdin>
  function adjustSeriesPosition() {
    const tocSidebar = document.getElementById("toc-sidebar");
    const seriesSidebar = document.getElementById("series-sidebar");
    if (!tocSidebar || !seriesSidebar) {
      return;
    }
    const sameSide = seriesSidebar.getAttribute("data-same-side") === "true";
    if (!sameSide) {
      return;
    }
    function updatePosition() {
      const tocHeight = tocSidebar.offsetHeight;
      const gap = 32;
      const topOffset = 96;
      const newTop = topOffset + tocHeight + gap;
      seriesSidebar.style.top = `${newTop}px`;
    }
    setTimeout(updatePosition, 100);
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(updatePosition, 200);
    });
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(updatePosition);
      observer.observe(tocSidebar);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", adjustSeriesPosition);
  } else {
    adjustSeriesPosition();
  }
})();
