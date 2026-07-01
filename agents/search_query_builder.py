from ML_core.core.schemas import SearchFeatureVector


class SearchQueryBuilder:
    def build_queries(self, vector: SearchFeatureVector) -> list[str]:
        terms = []
        if vector.destination:
            terms.append(vector.destination)
        if vector.property_type_hotel:
            terms.append("hotel")
        elif vector.property_type_resort:
            terms.append("resort")
        elif vector.property_type_apartment:
            terms.append("apartment")
        else:
            terms.append("room")
        if vector.near_beach:
            terms.append("near beach")
        if vector.amenity_pool:
            terms.append("pool")
        if vector.price_min_vnd or vector.price_max_vnd:
            terms.append(f"{vector.price_class_label} price")
        base_query = " ".join(terms)
        return [base_query, vector.raw_query] if vector.raw_query and vector.raw_query != base_query else [base_query]
