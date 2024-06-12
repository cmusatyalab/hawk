class MLClass:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.total_samples = 0

    def add_samples(self, count):
        if count < 0:
            raise ValueError("Count of samples cannot be negative.")
        self.total_samples += count

    def __repr__(self):
        return f"MLClass(name={self.name}, label={self.label}, total_samples={self.total_samples})"


class MLClassManager:
    def __init__(self):
        self.classes = {}
        self.class_list = []
        self.label_name_dict = {}

    def add_class(self, name, label):
        if name in self.classes:
            raise ValueError(f"Class {name} already exists.")
        new_class = MLClass(name, label)
        self.classes[name] = new_class
        self.class_list.append(new_class)
        self.label_name_dict[label] = name

    def get_class(self, name):
        return self.classes.get(name, None)

    def get_labels(self):
        return [str(cls.label) for cls in self.class_list]

    def add_samples(self, name, count):
        ml_class = self.get_class(name)
        if ml_class:
            ml_class.add_samples(count)
        else:
            raise ValueError(f"Class {name} does not exist.")

    def get_total_samples(self):
        return sum(ml_class.total_samples for ml_class in self.classes.values())

    def get_total_positives(self):
        return sum(
            ml_class.total_samples
            for ml_class in self.classes.values()
            if ml_class.label > 0
        )

    def __repr__(self):
        return f"MLClassManager(classes={list(self.classes.values())})"


"""
# Example usage
manager = MLClassManager()
manager.add_class("Negatives", 0)
manager.add_class("Roundabout", 1)
manager.add_class("Pool", 2)

# Add samples to the classes via manager
manager.add_samples("Negatives", 100)
manager.add_samples("Roundabout", 150)
manager.add_samples("Pool", 200)

print(manager)
# Output: MLClassManager(classes=[MLClass(name=Class1, label=0, total_samples=100), MLClass(name=Class2, label=1, total_samples=150), MLClass(name=Class3, label=2, total_samples=200)])

# Update samples
manager.add_samples("Negatives", 50)
print(manager.get_class("Negatives"))  # Output: MLClass(name=Class1, label=0, total_samples=150)

# Get total samples across all classes
total_samples = manager.get_total_samples()
print(f"Total samples across all classes: {total_samples}")  # Output: Total samples across all classes: 500

total_positive_samples = manager.get_total_positives()
print(f"Total positive samples: {total_positive_samples}")  # Output: Total positives samples across all classes: 500
"""
