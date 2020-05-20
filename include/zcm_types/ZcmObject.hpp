/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by zcm-gen
 **/

#include <zcm/zcm_coretypes.h>

#ifndef __ZcmObject_hpp__
#define __ZcmObject_hpp__

#include <string>
#include <vector>
#include "ZcmObjectGeometry.hpp"
#include "ZcmObjectGeometry.hpp"
#include "ZcmPoint.hpp"


/**
 * @struct ZcmObject
 * @brief Cодержит информацию о распознанном объекте
 * @var ZcmObject::bounding_box
 * @brief - описание геометрии объекта
 * @var ZcmObject::accuracy_box
 * @brief - точность (стандартное отклонение) для параметров, описывающих геометрию объекта
 * @var ZcmObject::detection_prob
 * @brief - вероятность обнаружения (принимает значения от 0 до 1)
 * @var ZcmObject::recognition_prob
 * @brief - вероятность распознавания (классификации) (принимает значения от 0 до 1)
 * @var ZcmObject::id_type
 * @brief - идентификатор типа объекта
 * @var ZcmObject::label_type
 * @brief - наименование типа объекта
 *                 ID : LABEL
 *                  0 : UNKNOWN       - неопределено
 *                  1 : SMALL         - маленький объект ( меньше 0.8 метров по ширине )
 *                  2 : MEDIUM        - объект среднего размера
 *                  3 : BIG           - большой объект ( больше 2 метров по ширине )
 *                  4 : PERSON        - человек
 *                  5 : CAR           - вагон
 *                  6 : TRAFFIC_LIGHT - светофор
 * 		    7 : MAP_OBJECT    - ствтичный объект карты
 * @var ZcmObject::contour_count
 * @brief - количество крайних точек объекта.
 * @var ZcmObject::contour
 * @brief - массив точек, характеризующий контур объекта (многоугольник).
 *
 */
class ZcmObject
{
    public:
        ZcmObjectGeometry bounding_box;

        ZcmObjectGeometry accuracy_box;

        double     detection_prob;

        double     recognition_prob;

        int32_t    id_type;

        std::string label_type;

        int32_t    contour_count;

        std::vector< ZcmPoint > contour;

    public:
        #if __cplusplus > 199711L /* if c++11 */
        static constexpr int8_t   OBSTACLE_TYPE_UNKNOWN = 0;
        static constexpr int8_t   OBSTACLE_TYPE_SMALL = 1;
        static constexpr int8_t   OBSTACLE_TYPE_MEDIUM = 2;
        static constexpr int8_t   OBSTACLE_TYPE_BIG = 3;
        static constexpr int8_t   OBSTACLE_TYPE_PERSON = 4;
        static constexpr int8_t   OBSTACLE_TYPE_CAR = 5;
        static constexpr int8_t   OBSTACLE_TYPE_TRAFFIC_LIGHT = 6;
        static constexpr int8_t   OBSTACLE_TYPE_MAP_OBJECT = 7;
        #else
        static const     int8_t   OBSTACLE_TYPE_UNKNOWN = 0;
        static const     int8_t   OBSTACLE_TYPE_SMALL = 1;
        static const     int8_t   OBSTACLE_TYPE_MEDIUM = 2;
        static const     int8_t   OBSTACLE_TYPE_BIG = 3;
        static const     int8_t   OBSTACLE_TYPE_PERSON = 4;
        static const     int8_t   OBSTACLE_TYPE_CAR = 5;
        static const     int8_t   OBSTACLE_TYPE_TRAFFIC_LIGHT = 6;
        static const     int8_t   OBSTACLE_TYPE_MAP_OBJECT = 7;
        #endif

    public:
        /**
         * Destructs a message properly if anything inherits from it
        */
        virtual ~ZcmObject() {}

        /**
         * Encode a message into binary form.
         *
         * @param buf The output buffer.
         * @param offset Encoding starts at thie byte offset into @p buf.
         * @param maxlen Maximum number of bytes to write.  This should generally be
         *  equal to getEncodedSize().
         * @return The number of bytes encoded, or <0 on error.
         */
        inline int encode(void* buf, uint32_t offset, uint32_t maxlen) const;

        /**
         * Check how many bytes are required to encode this message.
         */
        inline uint32_t getEncodedSize() const;

        /**
         * Decode a message from binary form into this instance.
         *
         * @param buf The buffer containing the encoded message.
         * @param offset The byte offset into @p buf where the encoded message starts.
         * @param maxlen The maximum number of bytes to reqad while decoding.
         * @return The number of bytes decoded, or <0 if an error occured.
         */
        inline int decode(const void* buf, uint32_t offset, uint32_t maxlen);

        /**
         * Retrieve the 64-bit fingerprint identifying the structure of the message.
         * Note that the fingerprint is the same for all instances of the same
         * message type, and is a fingerprint on the message type definition, not on
         * the message contents.
         */
        inline static int64_t getHash();

        /**
         * Returns "ZcmObject"
         */
        inline static const char* getTypeName();

        // ZCM support functions. Users should not call these
        inline int      _encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const;
        inline uint32_t _getEncodedSizeNoHash() const;
        inline int      _decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen);
        inline static uint64_t _computeHash(const __zcm_hash_ptr* p);
};

int ZcmObject::encode(void* buf, uint32_t offset, uint32_t maxlen) const
{
    uint32_t pos = 0;
    int thislen;
    int64_t hash = (int64_t)getHash();

    thislen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int ZcmObject::decode(const void* buf, uint32_t offset, uint32_t maxlen)
{
    uint32_t pos = 0;
    int thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

uint32_t ZcmObject::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t ZcmObject::getHash()
{
    static int64_t hash = _computeHash(NULL);
    return hash;
}

const char* ZcmObject::getTypeName()
{
    return "ZcmObject";
}

int ZcmObject::_encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const
{
    uint32_t pos = 0;
    int thislen;

    thislen = this->bounding_box._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->accuracy_box._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __double_encode_array(buf, offset + pos, maxlen - pos, &this->detection_prob, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __double_encode_array(buf, offset + pos, maxlen - pos, &this->recognition_prob, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->id_type, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    char* label_type_cstr = (char*) this->label_type.c_str();
    thislen = __string_encode_array(buf, offset + pos, maxlen - pos, &label_type_cstr, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->contour_count, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    for (int a0 = 0; a0 < this->contour_count; ++a0) {
        thislen = this->contour[a0]._encodeNoHash(buf, offset + pos, maxlen - pos);
        if(thislen < 0) return thislen; else pos += thislen;
    }

    return pos;
}

int ZcmObject::_decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen)
{
    uint32_t pos = 0;
    int thislen;

    thislen = this->bounding_box._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->accuracy_box._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __double_decode_array(buf, offset + pos, maxlen - pos, &this->detection_prob, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __double_decode_array(buf, offset + pos, maxlen - pos, &this->recognition_prob, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->id_type, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    int32_t __label_type_len__;
    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &__label_type_len__, 1);
    if(thislen < 0) return thislen; else pos += thislen;
    if((uint32_t)__label_type_len__ > maxlen - pos) return -1;
    this->label_type.assign(((const char*)buf) + offset + pos, __label_type_len__ - 1);
    pos += __label_type_len__;

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->contour_count, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    this->contour.resize(this->contour_count);
    for (int a0 = 0; a0 < this->contour_count; ++a0) {
        thislen = this->contour[a0]._decodeNoHash(buf, offset + pos, maxlen - pos);
        if(thislen < 0) return thislen; else pos += thislen;
    }

    return pos;
}

uint32_t ZcmObject::_getEncodedSizeNoHash() const
{
    uint32_t enc_size = 0;
    enc_size += this->bounding_box._getEncodedSizeNoHash();
    enc_size += this->accuracy_box._getEncodedSizeNoHash();
    enc_size += __double_encoded_array_size(NULL, 1);
    enc_size += __double_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += this->label_type.size() + 4 + 1;
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    for (int a0 = 0; a0 < this->contour_count; ++a0) {
        enc_size += this->contour[a0]._getEncodedSizeNoHash();
    }
    return enc_size;
}

uint64_t ZcmObject::_computeHash(const __zcm_hash_ptr* p)
{
    const __zcm_hash_ptr* fp;
    for(fp = p; fp != NULL; fp = fp->parent)
        if(fp->v == ZcmObject::getHash)
            return 0;
    const __zcm_hash_ptr cp = { p, (void*)ZcmObject::getHash };

    uint64_t hash = (uint64_t)0x5dafbc3e9fd12373LL +
         ZcmObjectGeometry::_computeHash(&cp) +
         ZcmObjectGeometry::_computeHash(&cp) +
         ZcmPoint::_computeHash(&cp);

    return (hash<<1) + ((hash>>63)&1);
}

#endif