class counted_follow_sets():
    
    # consecutive_items = list of nums
    def __init__(self, consecutive_items, conversion_dict=None):
        self.sets = dict()
        if conversion_dict is not None:
            canonical_app_names = []
            for item in consecutive_items:
                canonical_app_names.append(conversion_dict.get(item))
            consecutive_items = canonical_app_names

        if consecutive_items is not None and len(consecutive_items) > 0:
            for item, next_item in zip(consecutive_items[:-1], consecutive_items[1:]):
                self.add_item(item, next_item)
                
    def add_item(self, item, next_item):
        if (curr := self.sets.get(item)) is None:
            self.sets.update({item : dict({next_item : 1})})
        else:
            if (next_app_count := curr.get(next_item)) is None:
                curr.update({next_item : 1})
            else:
                curr.update({next_item : next_app_count+1})

        out_dict = {}
        for key, value in self.sets.items():
            sorted_vals = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))
            out_dict.update({key : sorted_vals})
        self.sets = out_dict

            
    def get_item_count(self, item, follow_item):
        if (ct := self.sets.get(item).get(follow_item)) is None:
            return 0
        return ct
    
    def unique_item_count(self):
        return len(self.sets)
    
    def is_in_set(self, item):
        return self.sets.get(item) is not None

    def get_sets(self):
        return self.sets