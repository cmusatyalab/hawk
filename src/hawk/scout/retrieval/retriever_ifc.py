from typing import Union, List, Any, Tuple


class RetrieverIfc:


    def get_ml_ready_data(self, object_ids: Union[List[Union[str, int]], Union[str, int]])\
            -> Union[List[Tuple[Any, Union[str, int]]], Tuple[Any, Union[str, int]]]:
        pass

    def get_oracle_ready_data(self, object_ids: Union[List[Union[str, int]], Union[str, int]])\
            -> Union[List[Tuple[Any, Union[str, int]]], Tuple[Any, Union[str, int]]]:
        pass

    def object_ids_stream(self)->Union[str,int]:  ## Generator
        pass

    def get_ground_truth(self, object_ids: Union[List[Union[str, int]],Union[str, int]]) \
            -> Union[List[Tuple[int, Union[str, int]]], Tuple[int, Union[str, int]]]:
        pass