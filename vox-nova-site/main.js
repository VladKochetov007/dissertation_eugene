// ========================================
// VOX NOVA — Main JavaScript
// ========================================

// Mobile Menu Toggle
function toggleMobileMenu() {
    const menu = document.querySelector('.mobile-menu');
    const toggle = document.querySelector('.nav-mobile-toggle');
    menu.classList.toggle('active');
    toggle.classList.toggle('active');
}

// Close mobile menu on link click
document.querySelectorAll('.mobile-menu a').forEach(link => {
    link.addEventListener('click', () => {
        document.querySelector('.mobile-menu').classList.remove('active');
        document.querySelector('.nav-mobile-toggle').classList.remove('active');
    });
});

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 50) {
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.06)';
    } else {
        navbar.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;

        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
            const navHeight = navbar.offsetHeight;
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Apply fade-in to sections
document.addEventListener('DOMContentLoaded', () => {
    const animatedElements = document.querySelectorAll(
        '.pipeline-step, .topic-card, .pillar-card, .stat-card, .team-card, .involve-card, .hub-detail-card, .eco-card, .output-card, .value-card'
    );

    animatedElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = `opacity 0.5s ease ${index % 5 * 0.1}s, transform 0.5s ease ${index % 5 * 0.1}s`;
        observer.observe(el);
    });
});
