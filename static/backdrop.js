// The page's backdrop takes the tint of the section crossing the middle of the screen (see .backdrop in style.css).

const tinted = new IntersectionObserver((entries) => {
  for (const entry of entries) if (entry.isIntersecting) document.body.dataset.tint = entry.target.dataset.tint;
}, { rootMargin: "-45% 0px -54% 0px" });

document.querySelectorAll("main > section[data-tint]").forEach((section) => tinted.observe(section));
