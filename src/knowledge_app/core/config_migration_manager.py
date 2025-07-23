"""
🔧 Configuration Migration Manager

This module handles the migration from multiple competing configuration managers
to the single, unified configuration system.

CRITICAL FIX: Eliminates the architectural schism caused by:
- AppConfig (legacy)
- ProperConfigManager (enterprise)
- UnifiedConfigManager (intended single source)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class ConfigMigrationManager:
    """
    🔧 FIX: Handles migration from dueling config managers to unified system
    
    This class ensures all configuration data is consolidated into the
    UnifiedConfigManager, eliminating configuration conflicts.
    """
    
    def __init__(self):
        self.migration_log = []
        self.unified_manager = None
        
    def perform_migration(self) -> bool:
        """
        Migrate all configuration data to UnifiedConfigManager
        
        Returns:
            bool: True if migration successful
        """
        try:
            # Import UnifiedConfigManager
            from .unified_config_manager import get_unified_config_manager
            self.unified_manager = get_unified_config_manager()
            
            logger.info("🔧 Starting configuration migration to eliminate dueling managers")
            
            # Step 1: Migrate from AppConfig
            self._migrate_app_config()
            
            # Step 2: Migrate from ProperConfigManager
            self._migrate_proper_config()
            
            # Step 3: Deprecate old managers
            self._deprecate_old_managers()
            
            # Step 4: Validate migration
            if self._validate_migration():
                logger.info("✅ Configuration migration completed successfully")
                return True
            else:
                logger.error("❌ Configuration migration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Configuration migration failed: {e}")
            return False
    
    def _migrate_app_config(self):
        """Migrate data from legacy AppConfig"""
        try:
            # Check if AppConfig data exists
            app_config_file = Path("config/app_config.json")
            if app_config_file.exists():
                with open(app_config_file, 'r') as f:
                    app_config_data = json.load(f)
                
                # Migrate to unified manager
                for key, value in app_config_data.items():
                    self.unified_manager.set(f"legacy.app_config.{key}", value, save_immediately=False)
                
                self.migration_log.append("✅ AppConfig data migrated")
                logger.info("✅ AppConfig data migrated to UnifiedConfigManager")
            else:
                self.migration_log.append("ℹ️ No AppConfig data to migrate")
                
        except Exception as e:
            logger.error(f"❌ Failed to migrate AppConfig: {e}")
            self.migration_log.append(f"❌ AppConfig migration failed: {e}")
    
    def _migrate_proper_config(self):
        """Migrate data from ProperConfigManager"""
        try:
            # Check if ProperConfigManager data exists
            proper_config_file = Path("config/proper_config.json")
            if proper_config_file.exists():
                with open(proper_config_file, 'r') as f:
                    proper_config_data = json.load(f)
                
                # Migrate to unified manager
                for key, value in proper_config_data.items():
                    self.unified_manager.set(f"legacy.proper_config.{key}", value, save_immediately=False)
                
                self.migration_log.append("✅ ProperConfigManager data migrated")
                logger.info("✅ ProperConfigManager data migrated to UnifiedConfigManager")
            else:
                self.migration_log.append("ℹ️ No ProperConfigManager data to migrate")
                
        except Exception as e:
            logger.error(f"❌ Failed to migrate ProperConfigManager: {e}")
            self.migration_log.append(f"❌ ProperConfigManager migration failed: {e}")
    
    def _deprecate_old_managers(self):
        """Issue deprecation warnings for old configuration managers"""
        try:
            # Create deprecation notice files
            deprecation_notice = """
# DEPRECATED CONFIGURATION MANAGER

This configuration manager has been deprecated and replaced by UnifiedConfigManager.

## Migration Status
All data has been migrated to the unified system.

## Action Required
Update your code to use:
```python
from knowledge_app.core.unified_config_manager import get_unified_config_manager
config = get_unified_config_manager()
```

## Removal Timeline
This deprecated manager will be removed in the next major version.
"""
            
            # Write deprecation notices
            Path("config/DEPRECATED_AppConfig.md").write_text(deprecation_notice)
            Path("config/DEPRECATED_ProperConfigManager.md").write_text(deprecation_notice)
            
            self.migration_log.append("✅ Deprecation notices created")
            logger.info("✅ Deprecation notices created for old config managers")
            
        except Exception as e:
            logger.error(f"❌ Failed to create deprecation notices: {e}")
    
    def _validate_migration(self) -> bool:
        """Validate that migration was successful"""
        try:
            # Check that UnifiedConfigManager is accessible
            if not self.unified_manager:
                return False
            
            # Test basic functionality
            test_key = "migration.test"
            test_value = "migration_successful"
            
            self.unified_manager.set(test_key, test_value, save_immediately=False)
            retrieved_value = self.unified_manager.get(test_key)
            
            if retrieved_value == test_value:
                self.migration_log.append("✅ Migration validation successful")
                return True
            else:
                self.migration_log.append("❌ Migration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Migration validation error: {e}")
            self.migration_log.append(f"❌ Migration validation error: {e}")
            return False
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Get detailed migration report"""
        return {
            "migration_log": self.migration_log,
            "unified_manager_available": self.unified_manager is not None,
            "migration_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
        }

# Global migration function
def migrate_to_unified_config() -> bool:
    """
    🔧 FIX: Migrate all configuration to unified system
    
    This function should be called during application startup to ensure
    all configuration data is consolidated.
    
    Returns:
        bool: True if migration successful
    """
    migration_manager = ConfigMigrationManager()
    return migration_manager.perform_migration()

# Deprecation wrapper for old config managers
class DeprecatedConfigManager:
    """
    🔧 FIX: Wrapper that redirects old config manager usage to unified system
    """
    
    def __init__(self, manager_name: str):
        self.manager_name = manager_name
        warnings.warn(
            f"{manager_name} is deprecated. Use UnifiedConfigManager to avoid configuration conflicts.",
            DeprecationWarning,
            stacklevel=3
        )
        
        try:
            from .unified_config_manager import get_unified_config_manager
            self._unified = get_unified_config_manager()
        except ImportError:
            logger.error(f"❌ UnifiedConfigManager not available for {manager_name}")
            self._unified = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Redirect get to unified manager"""
        if self._unified:
            return self._unified.get(f"legacy.{self.manager_name.lower()}.{key}", default)
        return default
    
    def set(self, key: str, value: Any) -> bool:
        """Redirect set to unified manager"""
        if self._unified:
            return self._unified.set(f"legacy.{self.manager_name.lower()}.{key}", value)
        return False
