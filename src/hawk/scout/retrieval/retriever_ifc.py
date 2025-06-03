from typing import Union, List, Any


class RetrieverIfc:


    def get_ml_ready_data(self, object_ids: Union[List[Union[str, int]], Union[str, int]])\
            -> Union[List[Any], Any]:
        pass

    def get_oracle_ready_data(self, object_ids: Union[List[Union[str, int]], Union[str, int]])\
            -> Union[List[Any], Any]:
        pass

    def object_ids_stream(self):  ## Generator
        pass