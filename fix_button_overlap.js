/**
 * Comprehensive Button Overlap Fix
 * Detects and fixes overlapping elements that prevent button clicks
 */

function fixButtonOverlap() {
    console.log('🔧 Starting comprehensive button overlap fix...');
    
    // Find all buttons that might be overlapped
    const buttons = document.querySelectorAll('button, .btn, [role="button"]');
    let fixedCount = 0;
    
    buttons.forEach((button, index) => {
        const buttonId = button.id || button.className || `button-${index}`;
        console.log(`🔍 Checking button: ${buttonId}`);
        
        const rect = button.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
            console.log(`⚠️ Button ${buttonId} has zero dimensions, skipping`);
            return;
        }
        
        // Check multiple points on the button
        const testPoints = [
            { x: rect.left + rect.width * 0.5, y: rect.top + rect.height * 0.5 }, // Center
            { x: rect.left + rect.width * 0.25, y: rect.top + rect.height * 0.25 }, // Top-left quarter
            { x: rect.left + rect.width * 0.75, y: rect.top + rect.height * 0.75 }, // Bottom-right quarter
        ];
        
        let isOverlapped = false;
        const overlappingElements = [];
        
        testPoints.forEach((point, pointIndex) => {
            const elementAtPoint = document.elementFromPoint(point.x, point.y);
            if (elementAtPoint && elementAtPoint !== button && !button.contains(elementAtPoint)) {
                isOverlapped = true;
                if (!overlappingElements.includes(elementAtPoint)) {
                    overlappingElements.push(elementAtPoint);
                }
            }
        });
        
        if (isOverlapped) {
            console.log(`⚠️ Button ${buttonId} is overlapped by:`, overlappingElements.map(el => ({
                tagName: el.tagName,
                id: el.id,
                className: el.className,
                zIndex: window.getComputedStyle(el).zIndex,
                position: window.getComputedStyle(el).position
            })));
            
            // Try to fix the overlap
            overlappingElements.forEach(overlappingEl => {
                const computedStyle = window.getComputedStyle(overlappingEl);
                
                // Strategy 1: Set pointer-events to none if it's not interactive
                if (!overlappingEl.onclick && !overlappingEl.addEventListener && 
                    overlappingEl.tagName !== 'BUTTON' && overlappingEl.tagName !== 'A' &&
                    !overlappingEl.hasAttribute('role')) {
                    
                    overlappingEl.style.pointerEvents = 'none';
                    console.log(`🔧 Set pointer-events: none on ${overlappingEl.tagName}#${overlappingEl.id}`);
                    fixedCount++;
                }
                
                // Strategy 2: Adjust z-index if needed
                else if (computedStyle.position !== 'static') {
                    const currentZIndex = parseInt(computedStyle.zIndex) || 0;
                    const buttonZIndex = parseInt(window.getComputedStyle(button).zIndex) || 0;
                    
                    if (currentZIndex >= buttonZIndex) {
                        button.style.zIndex = (currentZIndex + 10).toString();
                        console.log(`🔧 Increased button z-index to ${currentZIndex + 10}`);
                        fixedCount++;
                    }
                }
            });
            
            // Re-test after fixes
            const centerPoint = testPoints[0];
            const newElementAtPoint = document.elementFromPoint(centerPoint.x, centerPoint.y);
            if (newElementAtPoint === button) {
                console.log(`✅ Button ${buttonId} overlap fixed successfully`);
            } else {
                console.log(`⚠️ Button ${buttonId} still overlapped after fix attempt`);
            }
        } else {
            console.log(`✅ Button ${buttonId} is accessible`);
        }
    });
    
    console.log(`🔧 Button overlap fix complete. Fixed ${fixedCount} overlapping elements.`);
    return fixedCount;
}

// Auto-run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fixButtonOverlap);
} else {
    fixButtonOverlap();
}

// Export for manual use
window.fixButtonOverlap = fixButtonOverlap;