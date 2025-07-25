# Template project for Hawk plugins

## Retrievers

Retrievers are used by Hawk to fetch data from a data source. The data source
can be a file, a database, a web service, etc. The retriever is responsible for
fetching the data and returning it to Hawk.

There are 4 main functions a retriever needs to implement:

- `get_next_objectid() -> Iterator[ObjectId | None]`

   An iterator that returns object identifiers for individual objects. A lot of
   the Hawk retrievers tile a larger object into smaller items and the 'None'
   sentinel value is used to indicate when a larger object has been completely
   tiles. This is then used by Hawk to increment the 'images retrieved'
   statistic and introduce an optional per-object delay, which can be specified
   as a 'timeout' parameter in the retriever's configuration.

- `get_ml_data(objectid: ObjectId) -> HawkObject`

   Returns the Machine Learning ready data for a given object identifier. This
   data will be used by the configured machine learning model when inferencing
   or when building a labeled training dataset based on received user feedback.

   The HawkObject holds both the binary content as well as a media type field
   that describes the type of the binary content. It has class methods to create
   HawkObjects from files where it will attempt to guess the media type based
   on the file extension.

- `get_oracle_data(objectid: ObjectId) -> list[HawkObject]`

   Returns data for a given object identifier that will be transmitted to the
   Hawk home component. This data will be used to present the object to the user
   for labeling. The data can be anything that is useful for the user to label
   the object, e.g. an image, a video, a text, etc. The data is returned as a
   list of HawkObjects, which allows for multiple pieces of data to be
   transmitted to the user.

   The existing Hawk GUI web-browser based labeling tool will present the first
   item in a gallery view, and the remaining items will be shown in a popup
   when the user clicks on a 'View' button.

   If you are using your own labeling tool, you can use this function to return
   any data you want to be transmitted to the labeling tool, such as original
   raw sensor data.

- `get_groundtruth(objectid: ObjectId) -> list[Detection]`

   Returns the ground truth for a given object identifier. The ground truth is
   a list of detections, which are the bounding boxes and labels of known ground
   truth data for the object. This data is used for logging and statistical
   purposes. It is also passed along to the `script_labeler` tool, which is
   used to run automated missions that assume a 'perfect' labeling.

   A Detection consists of a `(center_x, center_y, width, height)` coordinate
   tuple which is normalized to the image size along with a class label string
   and an optional confidence score. For classification tasks, the bounding box
   coordinates are be set to `(0.5, 0.5, 1.0, 1.0)` to indicate the whole image
   and the confidence score can be set to `1.0` to indicate a perfect match.
