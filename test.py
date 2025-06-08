from Muffakir import MuffakirSearch
from Muffakir import MuffakirRAG
from Muffakir import MuffakirSyntheticData

config = {

        "llm_provider": "together",
        "api_key": "tgp_v1_tZD3diS9Qj7Y6__O4plxEoKsYQ06gQ0La4U1I_LFcIU",
        "data_dir": "test_qa",

    }

generator = MuffakirSyntheticData(config)
dataset = generator.generate_dataset(max_chunks=400)

