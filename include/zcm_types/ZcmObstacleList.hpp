/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by zcm-gen
 **/

#include <zcm/zcm_coretypes.h>

#ifndef __ZcmObstacleList_hpp__
#define __ZcmObstacleList_hpp__

#include <vector>
#include "ZcmService.hpp"
#include "ZcmObstacle.hpp"


/**
 * @struct ZcmObstacleList
 * @brief Cодержит информацию обо всех препятствиях в контролируемых зонах
 * @var ZcmObstacleList::service
 * @brief - служебное сообщение
 * @var ZcmObstacleList::obstacles_count
 * @brief - количество препятствий
 * @var ZcmObstacleList::obstacles
 * @brief - массив отслеживаемых препятствий
 * @var ZcmObstacleList::index_red
 * @brief - индекс ближайшего препятствия в красной зоне
 * @var ZcmObstacleList::index_yellow
 * @brief - индекс ближайшего препятствия в желтой зоне
 * @var ZcmObstacleList::index_brown
 * @brief - индекс ближайшего препятствия в коричневой зоне
 *
 */
class ZcmObstacleList
{
    public:
        ZcmService service;

        int32_t    obstacle_count;

        std::vector< ZcmObstacle > obstacles;

        int32_t    index_red;

        int32_t    index_yellow;

        int32_t    index_brown;

    public:
        /**
         * Destructs a message properly if anything inherits from it
        */
        virtual ~ZcmObstacleList() {}

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
         * Returns "ZcmObstacleList"
         */
        inline static const char* getTypeName();

        // ZCM support functions. Users should not call these
        inline int      _encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const;
        inline uint32_t _getEncodedSizeNoHash() const;
        inline int      _decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen);
        inline static uint64_t _computeHash(const __zcm_hash_ptr* p);
};

int ZcmObstacleList::encode(void* buf, uint32_t offset, uint32_t maxlen) const
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

int ZcmObstacleList::decode(const void* buf, uint32_t offset, uint32_t maxlen)
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

uint32_t ZcmObstacleList::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t ZcmObstacleList::getHash()
{
    static int64_t hash = _computeHash(NULL);
    return hash;
}

const char* ZcmObstacleList::getTypeName()
{
    return "ZcmObstacleList";
}

int ZcmObstacleList::_encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const
{
    uint32_t pos = 0;
    int thislen;

    thislen = this->service._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->obstacle_count, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    for (int a0 = 0; a0 < this->obstacle_count; ++a0) {
        thislen = this->obstacles[a0]._encodeNoHash(buf, offset + pos, maxlen - pos);
        if(thislen < 0) return thislen; else pos += thislen;
    }

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->index_red, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->index_yellow, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->index_brown, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int ZcmObstacleList::_decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen)
{
    uint32_t pos = 0;
    int thislen;

    thislen = this->service._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->obstacle_count, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    this->obstacles.resize(this->obstacle_count);
    for (int a0 = 0; a0 < this->obstacle_count; ++a0) {
        thislen = this->obstacles[a0]._decodeNoHash(buf, offset + pos, maxlen - pos);
        if(thislen < 0) return thislen; else pos += thislen;
    }

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->index_red, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->index_yellow, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->index_brown, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    return pos;
}

uint32_t ZcmObstacleList::_getEncodedSizeNoHash() const
{
    uint32_t enc_size = 0;
    enc_size += this->service._getEncodedSizeNoHash();
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    for (int a0 = 0; a0 < this->obstacle_count; ++a0) {
        enc_size += this->obstacles[a0]._getEncodedSizeNoHash();
    }
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    return enc_size;
}

uint64_t ZcmObstacleList::_computeHash(const __zcm_hash_ptr* p)
{
    const __zcm_hash_ptr* fp;
    for(fp = p; fp != NULL; fp = fp->parent)
        if(fp->v == ZcmObstacleList::getHash)
            return 0;
    const __zcm_hash_ptr cp = { p, (void*)ZcmObstacleList::getHash };

    uint64_t hash = (uint64_t)0x3848c6d4dee2a648LL +
         ZcmService::_computeHash(&cp) +
         ZcmObstacle::_computeHash(&cp);

    return (hash<<1) + ((hash>>63)&1);
}

#endif
