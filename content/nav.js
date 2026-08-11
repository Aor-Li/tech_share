/*
  Shared keyboard-paging controller for the AI Infra deck.
  Works identically whether it runs inside the merged index.html
  or inside a standalone chapter fragment opened on its own.
*/
(function () {
  function init() {
    var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
    if (slides.length === 0) return;

    var indicator = document.getElementById("page-indicator");
    var current = 0;

    function render() {
      slides.forEach(function (slide, i) {
        slide.classList.toggle("is-active", i === current);
      });
      if (indicator) {
        indicator.textContent = (current + 1) + " / " + slides.length;
      }
    }

    function go(delta) {
      var next = current + delta;
      if (next < 0 || next >= slides.length) return;
      current = next;
      render();
    }

    function goTo(index) {
      if (index < 0 || index >= slides.length) return;
      current = index;
      render();
    }
    window.deckGoTo = goTo;

    // Wire any element carrying data-goto to jump to that slide index.
    Array.prototype.forEach.call(
      document.querySelectorAll("[data-goto]"),
      function (el) {
        function jump() {
          goTo(parseInt(el.getAttribute("data-goto"), 10));
        }
        el.addEventListener("click", jump);
        el.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") {
            jump();
            e.preventDefault();
            e.stopPropagation();
          }
        });
      }
    );

    window.addEventListener("keydown", function (e) {
      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown":
        case "PageDown":
        case " ":
          go(1);
          e.preventDefault();
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "PageUp":
          go(-1);
          e.preventDefault();
          break;
        case "Home":
          current = 0;
          render();
          e.preventDefault();
          break;
        case "End":
          current = slides.length - 1;
          render();
          e.preventDefault();
          break;
      }
    });

    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
