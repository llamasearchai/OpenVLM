import pytest
from unittest.mock import MagicMock
from open_vlm.core.adapters.multi_adapter import (
    BaseAdapter,
    LoraAdapter,
    AdapterFactory
)

class TestBaseAdapter:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            BaseAdapter({})

class TestLoraAdapter:
    def test_initialization(self):
        config = {
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "lora_dropout": 0.1
        }
        adapter = LoraAdapter(config)
        assert adapter.peft_config.r == 8
        assert adapter.peft_config.target_modules == ["q_proj"]

    def test_adapt_model(self, mocker):
        mock_model = MagicMock()
        adapter = LoraAdapter({})
        mocker.patch('peft.get_peft_model', return_value="adapted_model")
        result = adapter.adapt(mock_model)
        assert result == "adapted_model"

class TestAdapterFactory:
    def test_registration(self):
        class TestAdapter(BaseAdapter):
            def adapt(self, model): return model
            def save(self, path): pass
            def load(self, path): pass
        
        AdapterFactory.register("test")(TestAdapter)
        assert "test" in AdapterFactory._registry
        
    def test_creation(self):
        adapter = AdapterFactory.create("lora", {})
        assert isinstance(adapter, LoraAdapter)

    def test_creation_invalid_type(self):
        with pytest.raises(ValueError):
            AdapterFactory.create("invalid", {})

    def test_duplicate_registration(self):
        with pytest.raises(ValueError):
            AdapterFactory.register("lora")(LoraAdapter)

class TestAdapterSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        config = {"lora_r": 8, "lora_alpha": 16}
        adapter = LoraAdapter(config)
        save_path = tmp_path / "adapter"
        adapter.save(save_path)
        
        new_adapter = LoraAdapter(config)
        new_adapter.load(save_path)
        assert new_adapter.peft_config.r == 8 