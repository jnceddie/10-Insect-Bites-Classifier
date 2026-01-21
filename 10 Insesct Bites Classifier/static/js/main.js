// Bug Bite Classifier - Main JavaScript

// Mobile Menu Toggle
document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const navMenu = document.getElementById('navMenu');

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Animation on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements with animation
    document.querySelectorAll('.feature-card, .bug-type-card, .about-card, .detail-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Disclaimer modal bindings (global)
    const globalDisclaimer = document.getElementById('disclaimerModal');
    const globalCheckbox = document.getElementById('disclaimerCheckbox');
    const globalProceed = document.getElementById('disclaimerProceed');
    const globalCancel = document.getElementById('disclaimerCancel');

    function showDisclaimer(event, href) {
        if (event) event.preventDefault();
        // store target href on proceed button
        globalProceed.dataset.targetHref = href || '/classifier';
        if (globalDisclaimer) globalDisclaimer.style.display = 'block';
        if (globalCheckbox) { globalCheckbox.checked = false; globalProceed.disabled = true; }
    }

    // Attach to primary start button if present
    const startBtn = document.getElementById('startClassifierBtn');
    if (startBtn) startBtn.addEventListener('click', (e) => showDisclaimer(e, '/classifier'));

    // Attach to any link or button that points to /classifier or has .requires-disclaimer
    document.querySelectorAll('a[href="/classifier"], a.requires-disclaimer, button.requires-disclaimer').forEach(el => {
        el.addEventListener('click', function(e) { showDisclaimer(e, this.getAttribute('href') || this.dataset.href); });
    });

    if (globalCheckbox) {
        globalCheckbox.addEventListener('change', () => {
            if (globalProceed) globalProceed.disabled = !globalCheckbox.checked;
        });
    }

    if (globalCancel) {
        globalCancel.addEventListener('click', () => {
            if (globalDisclaimer) globalDisclaimer.style.display = 'none';
            if (globalCheckbox) { globalCheckbox.checked = false; if (globalProceed) globalProceed.disabled = true; }
        });
    }

    if (globalProceed) {
        globalProceed.addEventListener('click', () => {
            const href = globalProceed.dataset.targetHref || '/classifier';
            window.location.href = href;
        });
    }

    // close global disclaimer when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === globalDisclaimer) {
            if (globalDisclaimer) globalDisclaimer.style.display = 'none';
            if (globalCheckbox) { globalCheckbox.checked = false; if (globalProceed) globalProceed.disabled = true; }
        }
    });
});

// Utility function to format class names
function formatClassName(className) {
    return className.replace('_', ' ')
                   .split(' ')
                   .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                   .join(' ');
}

// Utility function to validate image file
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    return validTypes.includes(file.type);
}

// Utility function to validate file size
function isValidFileSize(file, maxSizeMB = 16) {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
